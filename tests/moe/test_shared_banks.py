from __future__ import annotations

import torch

from freetoken.moe.shared_banks import SharedBankCache


def test_shared_bank_roundtrip(tmp_path, monkeypatch):
    """The cache maps the packed bytes back without an intermediate tensor copy."""
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    metadata = {
        "format_version": 1,
        "source": "synthetic",
        "num_layers": 2,
        "layout": {"gate_up": {"shape": [8], "dtype": "float16", "resident": False}},
    }
    cache = SharedBankCache(tmp_path / "banks", metadata)
    host, sources = cache.allocate({"gate_up": ((3, 8), torch.float16)}, 2)
    for layer_id, tensor in enumerate(sources["gate_up"]):
        tensor.fill_(layer_id + 1)
    alphas = {"gate_up_alpha": torch.arange(6, dtype=torch.float16)}

    with cache.exclusive():
        cache.finalize(host, sources, alphas, device=torch.device("cpu"))
    with cache.exclusive():
        loaded = cache.load(device=torch.device("cpu"))

    assert loaded is not None
    loaded_sources, loaded_alphas = loaded
    assert [float(t[0, 0]) for t in loaded_sources["gate_up"]] == [1.0, 2.0]
    assert torch.equal(loaded_alphas["gate_up_alpha"], alphas["gate_up_alpha"])
