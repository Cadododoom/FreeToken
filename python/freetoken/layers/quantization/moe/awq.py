"""GEMM-AWQ routed experts."""

from __future__ import annotations

import inspect

import torch

from freetoken.kernel import backend
from freetoken.utils import init_logger

from ..registry import LayerKind, register_method
from ..scheme import AWQ_GROUP, QuantKind
from .base import BankSpec, ExpertView, gated_epilogue_reason, MoEConfig, MoEKernel, MoEMethod

logger = init_logger(__name__)


def _marlin_symbols_ok() -> bool:
    """Probe the vLLM 0.14 AWQ-Marlin donor surface before loading banks."""
    try:
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (  # noqa: F401
            fused_marlin_moe,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # noqa: F401
            marlin_moe_permute_scales,
            moe_awq_to_marlin_zero_points,
        )

        if not hasattr(ops, "awq_marlin_moe_repack"):
            raise ImportError("vLLM _custom_ops.awq_marlin_moe_repack is missing")
    except Exception as exc:
        logger.warning(
            "AWQ Marlin backend is installed but unusable (%r); "
            "falling back to the reference AWQ backend",
            exc,
        )
        return False
    return True


def _awq_marlin_repack(qweight: torch.Tensor, *, size_k: int, size_n: int) -> torch.Tensor:
    """Call the vLLM 0.14 AWQ repacker across its minor API variants."""
    from vllm import _custom_ops as ops

    repack = ops.awq_marlin_moe_repack
    params = inspect.signature(repack).parameters
    if "perm" in params:
        # vLLM 0.14.x keeps an empty act-order tensor in this API even though AWQ
        # itself has no g_idx permutation.
        perm = torch.empty((qweight.shape[0], 0), dtype=torch.int32, device=qweight.device)
        return repack(qweight, perm, size_k=size_k, size_n=size_n, num_bits=4, is_a_8bit=False)
    return repack(qweight, size_k=size_k, size_n=size_n, num_bits=4, is_a_8bit=False)


class MarlinAwqMoEKernel(MoEKernel):
    """vLLM's SM80+ Marlin W4A16 kernel over AWQ-packed expert banks."""

    name = "marlin"
    max_slots = 992

    def unusable_reason(self, cfg: MoEConfig) -> str | None:
        reason = self._common_reject(
            cfg, resident_ok=False, tp_ok=True, cpu_ok=False, plain_silu_only=True
        )
        if reason:
            return reason
        if cfg.apply_router_weight_on_input and cfg.top_k != 1:
            return "AWQ Marlin input-side router weighting is only valid for topk=1"
        if cfg.hidden % 128 or cfg.local_intermediate % 64:
            return "AWQ Marlin requires hidden % 128 == 0 and TP-local intermediate % 64 == 0"
        if not backend.is_vllm_installed():
            return "vLLM is not installed"
        if not _marlin_symbols_ok():
            return "vLLM AWQ Marlin donor symbols are unusable"
        return None

    def worth_it(self, cfg: MoEConfig) -> bool:
        return (8, 0) <= backend.device_capability() < (10, 0)

    def layout(self, cfg: MoEConfig) -> dict[str, BankSpec]:
        i, h = cfg.local_intermediate, cfg.hidden
        return {
            # Marlin qweight is [K//16, N*2] int32 for unsigned INT4.
            "gate_up_qweight": BankSpec((h // 16, 4 * i), torch.int32),
            "gate_up_qzeros": BankSpec((h // AWQ_GROUP, 2 * i // 8), torch.int32),
            "gate_up_scales": BankSpec((h // AWQ_GROUP, 2 * i), torch.bfloat16),
            "down_qweight": BankSpec((i // 16, 2 * h), torch.int32),
            "down_qzeros": BankSpec((i // AWQ_GROUP, h // 8), torch.int32),
            "down_scales": BankSpec((i // AWQ_GROUP, h), torch.bfloat16),
        }

    @staticmethod
    def _fuse(parts: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        return torch.cat([parts[f"gate_{prefix}"], parts[f"up_{prefix}"]], dim=2)

    def pack(self, pieces, cfg: MoEConfig, out):
        """Repack raw GEMM-AWQ rows into Marlin's tiled W4A16 representation."""
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_moe_permute_scales,
            moe_awq_to_marlin_zero_points,
        )

        i, h = cfg.local_intermediate, cfg.hidden
        device = torch.device("cuda")
        gate_up_qweight = self._fuse(pieces, "qweight").to(device, non_blocking=True).contiguous()
        gate_up_qzeros = self._fuse(pieces, "qzeros").to(device, non_blocking=True).contiguous()
        gate_up_scales = self._fuse(pieces, "scales").to(device, non_blocking=True).contiguous()
        down_qweight = pieces["down_qweight"].to(device, non_blocking=True).contiguous()
        down_qzeros = pieces["down_qzeros"].to(device, non_blocking=True).contiguous()
        down_scales = pieces["down_scales"].to(device, non_blocking=True).contiguous()

        gate_up_qweight = _awq_marlin_repack(gate_up_qweight, size_k=h, size_n=2 * i)
        down_qweight = _awq_marlin_repack(down_qweight, size_k=i, size_n=h)
        gate_up_scales = marlin_moe_permute_scales(
            gate_up_scales, size_k=i, size_n=2 * i, group_size=AWQ_GROUP, is_a_8bit=False
        )
        down_scales = marlin_moe_permute_scales(
            down_scales, size_k=i, size_n=h, group_size=AWQ_GROUP, is_a_8bit=False
        )
        gate_up_qzeros = moe_awq_to_marlin_zero_points(
            gate_up_qzeros,
            size_k=gate_up_qzeros.shape[1],
            size_n=gate_up_qzeros.shape[2] * 8,
            num_bits=4,
            is_a_8bit=False,
        )
        down_qzeros = moe_awq_to_marlin_zero_points(
            down_qzeros,
            size_k=down_qzeros.shape[1],
            size_n=down_qzeros.shape[2] * 8,
            num_bits=4,
            is_a_8bit=False,
        )

        out["gate_up_qweight"].copy_(gate_up_qweight)
        out["gate_up_qzeros"].copy_(gate_up_qzeros)
        out["gate_up_scales"].copy_(gate_up_scales)
        out["down_qweight"].copy_(down_qweight)
        out["down_qzeros"].copy_(down_qzeros)
        out["down_scales"].copy_(down_scales)
        return {}

    def apply(self, layer, x, topk_weights, topk_ids, view: ExpertView, *, is_prefill: bool):
        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe
        from vllm.scalar_type import scalar_types

        t = view.tensors
        kwargs = {
            "hidden_states": x,
            "w1": t["gate_up_qweight"],
            "w2": t["down_qweight"],
            "bias1": None,
            "bias2": None,
            "w1_scale": t["gate_up_scales"],
            "w2_scale": t["down_scales"],
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "quant_type_id": scalar_types.uint4.id,
            "apply_router_weight_on_input": layer.apply_router_weight_on_input,
            "global_num_experts": view.n if view.n is not None else t["gate_up_qweight"].size(0),
            "activation": "silu",
            "w1_zeros": t["gate_up_qzeros"],
            "w2_zeros": t["down_qzeros"],
        }
        if "gating_output" in inspect.signature(fused_marlin_moe).parameters:
            kwargs["gating_output"] = None
        return fused_marlin_moe(**kwargs)


class ReferenceAwqMoEKernel(MoEKernel):
    """Correctness-first AWQ W4A16 path; Marlin can replace it later."""

    name = "reference"

    def unusable_reason(self, cfg: MoEConfig) -> str | None:
        reason = self._common_reject(
            cfg, resident_ok=False, tp_ok=True, cpu_ok=False, plain_silu_only=True
        )
        if reason:
            return reason
        reason = gated_epilogue_reason(cfg)
        if reason:
            return f"AWQ reference kernel: {reason}"
        if cfg.hidden % AWQ_GROUP or cfg.local_intermediate % AWQ_GROUP:
            return f"AWQ group size {AWQ_GROUP} must divide hidden and TP-local intermediate dimensions"
        if cfg.local_intermediate % 8:
            return "AWQ packed output columns require TP-local intermediate size divisible by 8"
        if cfg.apply_router_weight_on_input and cfg.beta != 0.0:
            return "AWQ reference kernel does not support beta with input-side router weighting"
        return None

    def layout(self, cfg: MoEConfig) -> dict[str, BankSpec]:
        i, h = cfg.local_intermediate, cfg.hidden
        return {
            "gate_up_qweight": BankSpec((h, 2 * i // 8), torch.int32),
            "gate_up_qzeros": BankSpec((h // AWQ_GROUP, 2 * i // 8), torch.int32),
            "gate_up_scales": BankSpec((h // AWQ_GROUP, 2 * i), torch.bfloat16),
            "down_qweight": BankSpec((i, h // 8), torch.int32),
            "down_qzeros": BankSpec((i // AWQ_GROUP, h // 8), torch.int32),
            "down_scales": BankSpec((i // AWQ_GROUP, h), torch.bfloat16),
        }

    @staticmethod
    def _fuse(parts: dict[str, torch.Tensor], prefix: str, dim: int) -> torch.Tensor:
        return torch.cat([parts[f"gate_{prefix}"], parts[f"up_{prefix}"]], dim=dim)

    def pack(self, pieces, cfg: MoEConfig, out):
        out["gate_up_qweight"].copy_(self._fuse(pieces, "qweight", dim=2))
        out["gate_up_qzeros"].copy_(self._fuse(pieces, "qzeros", dim=2))
        out["gate_up_scales"].copy_(self._fuse(pieces, "scales", dim=2))
        out["down_qweight"].copy_(pieces["down_qweight"])
        out["down_qzeros"].copy_(pieces["down_qzeros"])
        out["down_scales"].copy_(pieces["down_scales"])
        return {}

    def apply(self, layer, x, topk_weights, topk_ids, view: ExpertView, *, is_prefill: bool):
        from freetoken.moe.fused_awq import fused_experts_awq

        t = view.tensors
        return fused_experts_awq(
            x,
            t["gate_up_qweight"], t["gate_up_qzeros"], t["gate_up_scales"],
            t["down_qweight"], t["down_qzeros"], t["down_scales"],
            topk_weights, topk_ids, layer.activation, layer.apply_router_weight_on_input,
        )


@register_method(QuantKind.AWQ, LayerKind.MOE)
class AwqMoEMethod(MoEMethod):
    candidates = (MarlinAwqMoEKernel, ReferenceAwqMoEKernel)

    def create_weights(self, layer) -> None:
        raise NotImplementedError("AWQ experts are served from the offload cache, not resident")

    def resident_view(self, layer) -> ExpertView:
        raise NotImplementedError("AWQ experts are not resident")


__all__ = ["AwqMoEMethod", "MarlinAwqMoEKernel", "ReferenceAwqMoEKernel"]
