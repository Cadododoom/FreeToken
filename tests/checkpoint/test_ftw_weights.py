"""FTW replay: dropping entries by name before their bytes are read, and the vision-tower presence check."""

import torch

from freetoken.checkpoint.ftw import FTWReader, FTWWriter, ftw_tensor_names, iter_ftw_weights
from freetoken.models.weight import ftw_lacks_vision, load_weight
from freetoken.moe.expert_banks import load_expert_banks
from freetoken.moe.host_banks import HostBank


def _write_ftw(out_dir, names):
    writer = FTWWriter(str(out_dir))
    tensors = {name: torch.full((4, 8), float(i), dtype=torch.bfloat16) for i, name in enumerate(names)}
    for name, tensor in tensors.items():
        writer.add_tensor(name, tensor)
    writer.finalize({})
    return tensors


def test_keep_drops_entries_before_their_bytes_are_read(tmp_path, monkeypatch):
    tensors = _write_ftw(tmp_path, ["model.a.weight", "visual.b.weight", "model.c.weight"])
    read = []
    original = FTWReader.read_into

    def spy(self, dest, entry, **kwargs):
        read.append(entry["name"])
        return original(self, dest, entry, **kwargs)

    monkeypatch.setattr(FTWReader, "read_into", spy)
    got = dict(iter_ftw_weights(str(tmp_path), keep=lambda name: not name.startswith("visual.")))
    assert list(got) == ["model.a.weight", "model.c.weight"]
    assert read == ["model.a.weight", "model.c.weight"]
    for name, tensor in got.items():
        assert torch.equal(tensor, tensors[name])


def test_no_keep_replays_every_entry(tmp_path):
    tensors = _write_ftw(tmp_path, ["model.a.weight", "visual.b.weight"])
    got = dict(iter_ftw_weights(str(tmp_path)))
    assert list(got) == list(tensors)
    assert torch.equal(got["visual.b.weight"], tensors["visual.b.weight"])


def test_single_shard_ftw_region_maps_without_copy(tmp_path, monkeypatch):
    tensor = torch.arange(4 * 8, dtype=torch.bfloat16).view(4, 8)
    writer = FTWWriter(str(tmp_path))
    writer.add_tensor("gate_up#L00000", tensor, kind="experts_bank")
    index = writer.finalize({})
    entry = index["tensors"][0]

    reader = FTWReader(str(tmp_path))
    region = reader.single_shard_region(entry)
    assert region is not None
    path, offset = region
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    bank = HostBank.from_file_region(path, offset, tuple(tensor.shape), tensor.dtype)
    assert torch.equal(bank.tensor, tensor)
    reader.close()


def test_cross_shard_ftw_region_uses_copy_fallback(tmp_path):
    writer = FTWWriter(str(tmp_path), shard_limit=4096)
    tensor = torch.zeros(4096, dtype=torch.float32)
    writer.add_tensor("gate_up#L00000", tensor, kind="experts_bank")
    index = writer.finalize({})
    reader = FTWReader(str(tmp_path))
    assert reader.single_shard_region(index["tensors"][0]) is None
    reader.close()


def test_ftw_bank_loader_uses_file_backed_per_layer_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    gate_up = torch.arange(16, dtype=torch.bfloat16).view(2, 8)
    down = torch.arange(8, dtype=torch.bfloat16).view(2, 4)
    writer = FTWWriter(str(tmp_path))
    writer.add_tensor("gate_up#L00000", gate_up, kind="experts_bank")
    writer.add_tensor("down#L00000", down, kind="experts_bank")
    writer.finalize({"quant_format": "bf16", "expert_bank_num_layers": 1})

    class Config:
        num_moe_layers = 1

    banks = load_expert_banks(
        str(tmp_path), Config(), method=None, device=torch.device("cpu"),
        dtype=torch.bfloat16, workers=1,
    )
    assert banks.layer_residency == ["pinned"]
    assert torch.equal(banks.sources["gate_up"][0], gate_up)
    assert torch.equal(banks.sources["down"][0], down)


def test_load_weight_text_only_skips_the_tower(tmp_path):
    names = ["model.a.weight", "visual.b.weight", "vision_tower.c.weight"]
    _write_ftw(tmp_path, names)
    cpu = torch.device("cpu")
    assert [n for n, _ in load_weight(str(tmp_path), cpu, include_vision=False)] == ["model.a.weight"]
    assert [n for n, _ in load_weight(str(tmp_path), cpu)] == names


def test_ftw_lacks_vision(tmp_path):
    _write_ftw(tmp_path / "text", ["model.a.weight"])
    _write_ftw(tmp_path / "vl", ["model.a.weight", "visual.b.weight"])
    assert ftw_lacks_vision(str(tmp_path / "text"))
    assert not ftw_lacks_vision(str(tmp_path / "vl"))
    assert not ftw_lacks_vision(str(tmp_path))
    assert ftw_tensor_names(str(tmp_path / "vl"), "weight") == ["model.a.weight", "visual.b.weight"]
