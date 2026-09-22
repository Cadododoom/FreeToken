"""GLM-5.3 expert-parallel TP sharding helpers."""

import pytest
import torch

from freetoken.models.glm5_next.weight import _shard_fp8_expert


@pytest.mark.parametrize("proj", ["gate", "up"])
def test_fp8_gate_up_tp2_slices_rows(proj):
    weight = torch.arange(256 * 6, dtype=torch.float32).reshape(256, 6)
    scales = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3)

    assert torch.equal(
        _shard_fp8_expert(proj, "weight", weight, 256, 2, 1), weight[128:]
    )
    assert torch.equal(
        _shard_fp8_expert(proj, "weight_scale_inv", scales, 256, 2, 1),
        scales[1:],
    )


def test_fp8_down_tp2_slices_columns_and_scale_blocks():
    weight = torch.arange(6 * 8, dtype=torch.float32).reshape(6, 8)

    assert torch.equal(
        _shard_fp8_expert("down", "weight", weight, 8, 2, 1), weight[:, 4:]
    )
    # A 512-wide intermediate has four 128-wide scale blocks; rank 1 owns blocks 2:4.
    scale_grid = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
    assert torch.equal(
        _shard_fp8_expert("down", "weight_scale_inv", scale_grid, 512, 2, 1),
        scale_grid[:, 2:],
    )


def test_fp8_tp_requires_block_aligned_local_intermediate():
    with pytest.raises(ValueError, match="scale grid"):
        _shard_fp8_expert(
            "gate", "weight_scale_inv", torch.ones(3, 4), 384, 2, 1
        )
