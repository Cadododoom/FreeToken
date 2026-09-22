"""Deployment-time quantization for dense weights left unquantized by a checkpoint.

``QuantConfig`` normally describes what a checkpoint stores.  This wrapper describes a
deployment choice instead: convert eligible BF16 dense projections to per-output-row FP8
at load time, then serve them through the existing W8A16 Triton kernel.  Routed experts
and small routing/state projections remain on their checkpoint format.
"""

from __future__ import annotations

from typing import Any

from .configs.base import NoQuantConfig, QuantConfig
from .linear import LinearConfig
from .names import is_routed_expert, name_set
from .quant_backend import get_quant_backend
from .registry import LayerKind, method_class
from .scheme import QuantKind, QuantScheme, fp8_tensor_scheme


AT_LOAD_FP8 = fp8_tensor_scheme("float", per_row=True)

# These projections are small or numerically sensitive control/state paths.  Keeping them
# BF16 preserves routing and recurrent-state stability while still quantizing the large
# attention, shared-MLP, embedding, and language-head matrices.
KEEP_BF16 = name_set((
    "*.gate", "*.shared_expert_gate", "*hyper_connection*", "*.indexer", "*.ple",
    "*.in_proj_ba", "*.in_proj_b", "*.in_proj_a",
    "visual", "*.visual",
))


class LoadTimeFp8Config(QuantConfig):
    """Expose eligible unquantized linear modules as per-row FP8 model buffers."""

    dialect = "dense-quant-fp8"

    def __init__(self, inner: QuantConfig | None):
        inner = inner if inner is not None else NoQuantConfig()
        super().__init__(inner.name_map, ())
        self.inner = inner

    @classmethod
    def claims(cls, q: dict[str, Any]) -> bool:
        return False  # selected only by the explicit --dense-quant flag

    def scheme_for_name(self, name: str) -> QuantScheme | None:
        return self.inner.scheme_for_name(name)

    def _quantize_at_load(self, prefix: str) -> bool:
        if self.inner.scheme_for(prefix) is not None:
            return False  # native checkpoint quantization wins
        names = self.name_map.to_checkpoint(prefix)
        return not any(KEEP_BF16(n) or is_routed_expert(n) for n in names)

    def scheme_for(self, prefix: str) -> QuantScheme | None:
        inner = self.inner.scheme_for(prefix)
        if inner is not None:
            return inner
        return AT_LOAD_FP8 if self._quantize_at_load(prefix) else None

    def get_quant_method(self, layer: Any, prefix: str):
        if layer.quant_layer_kind is not LayerKind.LINEAR or not self._quantize_at_load(prefix):
            return self.inner.get_quant_method(layer, prefix)
        cls = method_class(QuantKind.FP8_TENSOR, LayerKind.LINEAR)
        cfg = LinearConfig.from_layer(layer, AT_LOAD_FP8)
        return cls(cfg, get_quant_backend().select(LayerKind.LINEAR, QuantKind.FP8_TENSOR))
