"""Reference GEMM-AWQ W4A16 MoE execution.

The first AWQ path intentionally keeps the unpack/dequant logic in PyTorch. It is
not the performance backend: it gives the TP/offload loader a numerically auditable
execution path before an optional Marlin repack is added.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


_AWQ_REVERSE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def unpack_awq(packed: torch.Tensor) -> torch.Tensor:
    """Unpack GEMM-AWQ int32 words and restore logical output-column order."""
    if packed.dtype is not torch.int32:
        raise TypeError(f"AWQ packed tensors must be int32, got {packed.dtype}")
    shifts = torch.arange(0, 32, 4, dtype=packed.dtype, device=packed.device)
    values = torch.bitwise_and(
        torch.bitwise_right_shift(packed.unsqueeze(-1), shifts), 0xF
    )
    values = values.reshape(*packed.shape[:-1], -1, 8)
    order = torch.tensor(_AWQ_REVERSE_ORDER, dtype=torch.long, device=packed.device)
    values = values.index_select(-1, order)
    return values.reshape(*packed.shape[:-1], -1)


def dequant_awq(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 32,
) -> torch.Tensor:
    """Return logical ``[experts, K, N]`` W4A16 weights."""
    codes = unpack_awq(qweight).to(torch.float32)
    zeros = unpack_awq(qzeros).to(torch.float32)
    scales = scales.to(torch.float32)
    if codes.shape[1] != zeros.shape[1] * group_size:
        raise ValueError(
            f"AWQ qweight K={codes.shape[1]} does not match "
            f"qzeros groups={zeros.shape[1]} x group_size={group_size}"
        )
    zeros = zeros.repeat_interleave(group_size, dim=1)
    scales = scales.repeat_interleave(group_size, dim=1)
    return (codes - zeros) * scales


@torch.no_grad()
def fused_experts_awq(
    hidden_states: torch.Tensor,
    gate_up_qweight: torch.Tensor,
    gate_up_qzeros: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_qweight: torch.Tensor,
    down_qzeros: torch.Tensor,
    down_scales: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    apply_router_weight_on_input: bool,
) -> torch.Tensor:
    """Reference W4A16 grouped MoE over cache slots or a materialized layer."""
    if activation != "silu":
        raise ValueError(f"AWQ reference kernel supports silu only, got {activation!r}")
    tokens, top_k = topk_ids.shape
    flat_ids = topk_ids.reshape(-1).to(torch.long)
    route_weights = topk_weights.reshape(-1).to(torch.float32)
    route_hidden = hidden_states.repeat_interleave(top_k, dim=0).to(torch.float32)
    if apply_router_weight_on_input:
        route_hidden = route_hidden * route_weights[:, None]

    gate_up = dequant_awq(
        gate_up_qweight.index_select(0, flat_ids),
        gate_up_qzeros.index_select(0, flat_ids),
        gate_up_scales.index_select(0, flat_ids),
    )
    gate_up_out = torch.bmm(route_hidden.unsqueeze(1), gate_up).squeeze(1)
    half = gate_up_out.shape[1] // 2
    intermediate = F.silu(gate_up_out[:, :half]) * gate_up_out[:, half:]

    down = dequant_awq(
        down_qweight.index_select(0, flat_ids),
        down_qzeros.index_select(0, flat_ids),
        down_scales.index_select(0, flat_ids),
    )
    routed = torch.bmm(intermediate.unsqueeze(1), down).squeeze(1)
    if not apply_router_weight_on_input:
        routed = routed * route_weights[:, None]
    return routed.reshape(tokens, top_k, -1).sum(dim=1).to(hidden_states.dtype)


__all__ = ["dequant_awq", "fused_experts_awq", "unpack_awq"]
