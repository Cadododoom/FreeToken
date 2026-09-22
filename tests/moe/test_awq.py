"""Correctness and layout tests for the initial GEMM-AWQ MoE path."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.layers.quantization.moe.awq import MarlinAwqMoEKernel, ReferenceAwqMoEKernel
from freetoken.layers.quantization.moe.base import ExpertView, MoEConfig
from freetoken.moe.fused_awq import dequant_awq, unpack_awq


def _pack_awq(values: torch.Tensor) -> torch.Tensor:
    raw_order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], dtype=torch.long)
    values = values.reshape(*values.shape[:-1], values.shape[-1] // 8, 8).index_select(-1, raw_order)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    return (values.to(torch.int32) << shifts).sum(dim=-1).to(torch.int32)


def test_awq_unpack_restores_logical_column_order():
    logical = torch.arange(32 * 16, dtype=torch.int32).reshape(32, 16).remainder(16)
    packed = _pack_awq(logical)
    assert torch.equal(unpack_awq(packed), logical)
    zeros = torch.zeros(1, 2, dtype=torch.int32)
    scales = torch.ones(1, 16, dtype=torch.bfloat16)
    assert torch.equal(dequant_awq(packed.unsqueeze(0), zeros.unsqueeze(0), scales.unsqueeze(0))[0], logical.float())


def test_awq_tp2_layout_and_reference_apply():
    experts, hidden, local_intermediate = 2, 64, 32
    cfg = MoEConfig(
        num_experts=experts,
        hidden=hidden,
        intermediate=hidden,
        top_k=2,
        tp_size=2,
        strategy="offload",
    )
    kernel = ReferenceAwqMoEKernel()
    layout = kernel.layout(cfg)
    assert layout["gate_up_qweight"].shape == (hidden, 2 * local_intermediate // 8)
    assert layout["down_qweight"].shape == (local_intermediate, hidden // 8)

    pieces = {}
    for proj, out_features, in_features in (
        ("gate", local_intermediate, hidden),
        ("up", local_intermediate, hidden),
        ("down", hidden, local_intermediate),
    ):
        pieces[f"{proj}_qweight"] = _pack_awq(torch.randint(0, 16, (experts, in_features, out_features)))
        pieces[f"{proj}_qzeros"] = _pack_awq(torch.randint(0, 16, (experts, in_features // 32, out_features)))
        pieces[f"{proj}_scales"] = torch.rand(experts, in_features // 32, out_features).bfloat16()
    out = {name: torch.empty((experts, *spec.shape), dtype=spec.dtype) for name, spec in layout.items()}
    kernel.pack(pieces, cfg, out)

    layer = SimpleNamespace(activation="silu", apply_router_weight_on_input=False)
    x = torch.randn(2, hidden, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    weights = torch.tensor([[0.4, 0.6], [0.7, 0.3]])
    got = kernel.apply(layer, x, weights, ids, ExpertView(out), is_prefill=False)

    gate_up = dequant_awq(out["gate_up_qweight"], out["gate_up_qzeros"], out["gate_up_scales"])
    down = dequant_awq(out["down_qweight"], out["down_qzeros"], out["down_scales"])
    expected = []
    for row in range(x.shape[0]):
        routed = []
        for col in range(ids.shape[1]):
            expert = int(ids[row, col])
            gate_up_out = x[row].float() @ gate_up[expert]
            intermediate = torch.nn.functional.silu(gate_up_out[:local_intermediate]) * gate_up_out[local_intermediate:]
            routed.append((intermediate @ down[expert]) * weights[row, col])
        expected.append(sum(routed))
    assert torch.equal(got, torch.stack(expected).bfloat16())


def test_awq_marlin_tp2_layout():
    cfg = MoEConfig(
        num_experts=512,
        hidden=2560,
        intermediate=640,
        top_k=10,
        tp_size=2,
        strategy="offload",
    )
    layout = MarlinAwqMoEKernel().layout(cfg)
    assert layout["gate_up_qweight"].shape == (160, 1280)
    assert layout["gate_up_qzeros"].shape == (80, 80)
    assert layout["down_qweight"].shape == (20, 5120)
    assert layout["down_qzeros"].shape == (10, 320)
