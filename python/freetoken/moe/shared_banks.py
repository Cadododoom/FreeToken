"""Optional read-only, file-backed expert-bank sharing between serve processes.

An offload endpoint normally packs the routed experts into anonymous host mmaps.  That is
fast, but two independent endpoints each own a full copy of the bank.  This module adds a
conservative opt-in cache: the first process packs directly into aligned shared tmpfs files;
later processes map those same files. Linux therefore keeps one physical page set while every
process gets its own CUDA host registration and GPU-visible address.

The cache is deliberately not a general persistence format.  It is keyed by the source
checkpoint, expert kernel layout, and TP rank, and its manifest is published last.  A crash
before publication leaves no loadable cache and the next process rebuilds it under the lock.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import mmap
import os
import re
from pathlib import Path

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

_FORMAT_VERSION = 1
_ALIGN = 4096
_CHUNK = 8 << 20
_MANIFEST = "manifest.json"
_LOCK = ".build.lock"
_SHARED_PROBES: set[str] = set()


class SharedBankError(RuntimeError):
    """The requested shared bank cache cannot be safely created or loaded."""


def _align_up(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_of(name: str) -> torch.dtype:
    return getattr(torch, name)


def _write_direct(path: Path, source: memoryview) -> None:
    """Write an aligned buffer without populating a second page-cache copy."""
    nbytes = len(source)
    if nbytes % _ALIGN:
        raise SharedBankError(f"shared bank buffer is not aligned: {nbytes} bytes")
    direct = getattr(os, "O_DIRECT", 0)
    if not direct:
        raise SharedBankError("shared expert banks require O_DIRECT on this platform")
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | direct
    try:
        fd = os.open(tmp, flags, 0o600)
    except OSError as exc:
        raise SharedBankError(
            f"cannot create O_DIRECT shared expert bank {path}: {exc}; "
            "use a filesystem that supports direct I/O or omit --moe-shared-bank-dir"
        ) from exc
    try:
        os.ftruncate(fd, nbytes)
        offset = 0
        while offset < nbytes:
            end = min(offset + _CHUNK, nbytes)
            # HostBank buffers are page aligned and the chunk is a 4096-byte multiple.
            done = 0
            while done < end - offset:
                got = os.pwrite(fd, source[offset + done:end], offset + done)
                if got <= 0:
                    raise OSError(f"short write at {offset + done} ({got} bytes)")
                done += got
            offset = end
        os.fsync(fd)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass
    except BaseException:
        try:
            os.close(fd)
        finally:
            tmp.unlink(missing_ok=True)
        raise
    else:
        os.close(fd)
        os.replace(tmp, path)


class SharedBankCache:
    """One model/layout-specific shared host-bank cache."""

    def __init__(self, root: str | os.PathLike[str], metadata: dict):
        self.root = Path(root)
        self.metadata = metadata

    @classmethod
    def for_method(cls, base_dir: str, model_path: str, model_config, method):
        """Select a stable cache directory for one source/layout/TP combination."""
        from freetoken.distributed import try_get_tp_info

        source = os.path.realpath(model_path) if os.path.isdir(model_path) else str(model_path)
        config_stat = None
        config_path = os.path.join(model_path, "config.json")
        try:
            st = os.stat(config_path)
            config_stat = [st.st_size, st.st_mtime_ns]
        except OSError:
            pass
        tp = try_get_tp_info()
        layout = {
            role: {
                "shape": list(spec.shape),
                "dtype": _dtype_name(spec.dtype),
                "resident": bool(spec.resident),
            }
            for role, spec in method.layout().items()
        }
        metadata = {
            "format_version": _FORMAT_VERSION,
            "source": source,
            "config_stat": config_stat,
            "kind": str(method.kind),
            "kernel": method.kernel.name,
            "num_layers": int(model_config.num_moe_layers),
            "num_experts": int(method.cfg.num_experts),
            "tp": None if tp is None else [int(tp.rank), int(tp.size)],
            "layout": layout,
        }
        digest = hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:20]
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(source).name or "model")
        return cls(Path(base_dir).expanduser() / f"{safe}-{digest}", metadata)

    @contextlib.contextmanager
    def exclusive(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / _LOCK, "a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _manifest(self) -> dict | None:
        try:
            with open(self.root / _MANIFEST, encoding="utf-8") as f:
                manifest = json.load(f)
        except (FileNotFoundError, OSError, ValueError):
            return None
        if manifest.get("metadata") != self.metadata:
            return None
        for item in manifest.get("banks", []):
            try:
                if os.path.getsize(self.root / item["file"]) != int(item["nbytes"]):
                    return None
            except (KeyError, OSError, TypeError, ValueError):
                return None
        for item in manifest.get("alphas", []):
            try:
                if os.path.getsize(self.root / item["file"]) != int(item["nbytes"]):
                    return None
            except (KeyError, OSError, TypeError, ValueError):
                return None
        return manifest

    def _ensure_shared_mapping(self) -> None:
        """Fail before a large build when the filesystem cannot pin shared file pages.

        CUDA's long-term page pinning rejects shared ext4 file mappings on this host, while
        shared tmpfs mappings are both registerable and physically shareable.  A 4-KiB probe
        makes that distinction without committing any model-sized allocation.  In practice
        this means ``/dev/shm`` (which can be much larger than the conventional 64-MiB tmpfs
        default) is the correct location for a shared pinned bank cache.
        """
        key = str(self.root)
        if key in _SHARED_PROBES:
            return
        # The converter and CPU-only regression tests intentionally disable CUDA bank
        # registration.  They still exercise the file-backed mapping and manifest path;
        # serving never sets this switch because the GPU transfer path needs registration.
        if os.environ.get("FREETOKEN_SKIP_BANK_PIN", "").strip().lower() in ("1", "true", "yes", "on"):
            self.root.mkdir(parents=True, exist_ok=True)
            _SHARED_PROBES.add(key)
            return
        from freetoken.moe.host_banks import HostBank

        self.root.mkdir(parents=True, exist_ok=True)
        probe = self.root / f".cuda-shared-probe-{os.getpid()}"
        try:
            with open(probe, "wb") as f:
                f.truncate(_ALIGN)
            bank = HostBank.from_file(str(probe), (_ALIGN,), torch.uint8)
            bank.pin()
        except Exception as exc:  # noqa: BLE001 -- driver/filesystem-specific probe
            probe.unlink(missing_ok=True)
            raise SharedBankError(
                f"CUDA cannot register shared file pages under {self.root}; "
                "use a large tmpfs path such as /dev/shm/freetoken-shared-banks "
                "for --moe-shared-bank-dir"
            ) from exc
        probe.unlink(missing_ok=True)
        _SHARED_PROBES.add(key)

    @staticmethod
    def _bank_name(role: str, layer_id: int) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", role)
        return f"bank-{safe}-L{layer_id:05d}.bin"

    @staticmethod
    def _alpha_name(role: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", role)
        return f"alpha-{safe}.bin"

    def load(self, *, device: torch.device):
        """Map and pin a complete cache, returning ``(sources, alphas)`` or ``None``."""
        manifest = self._manifest()
        if manifest is None:
            return None
        self._ensure_shared_mapping()
        from freetoken.moe.host_banks import HostBank

        sources: dict[str, list[torch.Tensor]] = {}
        for item in manifest["banks"]:
            role = item["role"]
            layer_id = int(item["layer"])
            bank = HostBank.from_file(
                str(self.root / item["file"]),
                tuple(item["shape"]),
                _dtype_of(item["dtype"]),
            )
            bank.pin()
            sources.setdefault(role, []).append((layer_id, bank.tensor))
        normalized = {}
        for role, entries in sources.items():
            normalized[role] = [tensor for _, tensor in sorted(entries)]

        alphas = {}
        for item in manifest.get("alphas", []):
            dtype = _dtype_of(item["dtype"])
            numel = int(item["nbytes"]) // torch.empty((), dtype=dtype).element_size()
            alpha = torch.from_file(
                str(self.root / item["file"]), shared=False, size=numel, dtype=dtype
            ).view(*item["shape"])
            alphas[item["role"]] = alpha.to(device=device, non_blocking=True)
        return normalized, alphas

    def allocate(self, specs, num_layers: int):
        """Create the shared bank files and map them before the first pack.

        Building directly into shared tmpfs mappings is important: allocating anonymous
        banks and copying them into tmpfs would briefly require two full host-bank copies.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        self._ensure_shared_mapping()
        from freetoken.moe.host_banks import HostBank

        host_banks = {}
        sources = {}
        for role, (shape, dtype) in specs.items():
            per_host = []
            per_source = []
            nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            asize = _align_up(nbytes)
            for layer_id in range(num_layers):
                path = self.root / self._bank_name(role, layer_id)
                with open(path, "w+b") as f:
                    f.truncate(asize)
                bank = HostBank.from_file(str(path), shape, dtype)
                per_host.append(bank)
                per_source.append(bank.tensor)
            host_banks[role] = per_host
            sources[role] = per_source
        return host_banks, sources

    def finalize(self, host_banks, sources, alphas, *, device: torch.device):
        """Pin a directly-built shared bank set, write alpha sidecars, and publish the manifest."""
        required = sum(len(bank.memoryview()) for per_role in host_banks.values() for bank in per_role)
        free = os.statvfs(self.root).f_bavail * os.statvfs(self.root).f_frsize
        if free < required + (8 << 30):
            raise SharedBankError(
                f"shared expert bank cache needs {required / 2**30:.1f} GiB plus an 8 GiB margin, "
                f"but only {free / 2**30:.1f} GiB is free on {self.root}"
            )
        manifest = {"format": "freetoken_shared_expert_banks", "metadata": self.metadata,
                    "banks": [], "alphas": []}
        for role, per_layer in sources.items():
            for layer_id, tensor in enumerate(per_layer):
                bank = host_banks[role][layer_id]
                bank.pin()
                manifest["banks"].append({
                    "role": role, "layer": layer_id,
                    "file": self._bank_name(role, layer_id),
                    "shape": list(tensor.shape), "dtype": _dtype_name(tensor.dtype),
                    "nbytes": len(bank.memoryview()),
                })

        for role, alpha in alphas.items():
            filename = self._alpha_name(role)
            path = self.root / filename
            cpu = alpha.detach().to(device="cpu").contiguous().view(torch.uint8)
            raw = cpu.numpy().tobytes()
            tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            with open(tmp, "wb") as f:
                f.write(raw)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            manifest["alphas"].append({
                "role": role, "file": filename, "shape": list(alpha.shape),
                "dtype": _dtype_name(alpha.dtype), "nbytes": len(raw),
            })

        tmp = self.root / f".{_MANIFEST}.tmp-{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.root / _MANIFEST)
        logger.info_rank0(f"shared expert bank cache ready: {self.root}")
        return sources, alphas

    def publish(self, host_banks, sources, alphas, *, device: torch.device):
        """Write one complete cache and replace anonymous sources with mapped sources."""
        self.root.mkdir(parents=True, exist_ok=True)
        required = sum(len(bank.memoryview()) for per_role in host_banks.values() for bank in per_role)
        try:
            free = os.statvfs(self.root).f_bavail * os.statvfs(self.root).f_frsize
        except OSError as exc:
            raise SharedBankError(f"cannot inspect free space for shared bank cache {self.root}: {exc}") from exc
        # Leave a generous margin for the model files and normal runtime writes.  A cache
        # build that would consume the disk is worse than a clean startup refusal.
        if free < required + (8 << 30):
            raise SharedBankError(
                f"shared expert bank cache needs {required / 2**30:.1f} GiB plus an 8 GiB margin, "
                f"but only {free / 2**30:.1f} GiB is free on {self.root}"
            )
        manifest = {"format": "freetoken_shared_expert_banks", "metadata": self.metadata,
                    "banks": [], "alphas": []}

        # Write and remap one bank at a time.  ``release`` drops the anonymous pages after
        # the direct write, so the conversion never needs a second 68-GiB resident copy.
        mapped: dict[str, list[torch.Tensor]] = {role: [] for role in sources}
        for role, per_layer in sources.items():
            for layer_id, tensor in enumerate(per_layer):
                bank = host_banks[role][layer_id]
                filename = self._bank_name(role, layer_id)
                path = self.root / filename
                _write_direct(path, bank.memoryview())
                bank.release()
                from freetoken.moe.host_banks import HostBank

                replacement = HostBank.from_file(str(path), tuple(tensor.shape), tensor.dtype)
                replacement.pin()
                host_banks[role][layer_id] = replacement
                mapped[role].append(replacement.tensor)
                manifest["banks"].append({
                    "role": role, "layer": layer_id, "file": filename,
                    "shape": list(tensor.shape), "dtype": _dtype_name(tensor.dtype),
                    "nbytes": len(bank.memoryview()),
                })

        for role, alpha in alphas.items():
            filename = self._alpha_name(role)
            path = self.root / filename
            cpu = alpha.detach().to(device="cpu").contiguous().view(torch.uint8)
            raw = cpu.numpy().tobytes()
            # Alpha vectors are tiny; ordinary buffered I/O avoids imposing an alignment
            # requirement on a temporary Python byte buffer.  The large expert banks above
            # are the RAM-critical part and always use O_DIRECT.
            tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            with open(tmp, "wb") as f:
                f.write(raw)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            alpha_item = {
                "role": role, "file": filename, "shape": list(alpha.shape),
                "dtype": _dtype_name(alpha.dtype), "nbytes": len(raw),
            }
            manifest["alphas"].append(alpha_item)

        tmp = self.root / f".{_MANIFEST}.tmp-{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.root / _MANIFEST)
        logger.info_rank0(f"shared expert bank cache ready: {self.root}")
        # The newly written files are already registered; return the mapped tensors and the
        # original alpha tensors.  A later process reconstructs alphas from the small files.
        return mapped, alphas


__all__ = ["SharedBankCache", "SharedBankError"]
