from __future__ import annotations

from typing import Any, ClassVar

from ..names import is_routed_expert, name_set
from ..registry import register_dialect
from ..scheme import AWQ_GROUP, QuantKind, QuantScheme, awq_scheme
from .base import QuantConfig, Stored


@register_dialect
class AwqConfig(QuantConfig):
    """GEMM-AWQ checkpoints with asymmetric per-group routed-expert weights.

    The first supported AWQ family is Qwen4Exp's per-expert W4A16 export. Dense AWQ
    linears are deliberately left unquantized until they have a separate linear
    kernel and loader; silently treating those tensors as supported would corrupt a
    mixed checkpoint.
    """

    dialect = "awq"
    SCHEME: ClassVar[QuantScheme] = awq_scheme()
    STORAGE: ClassVar[dict[QuantKind, dict[str, str | Stored]]] = {
        QuantKind.AWQ: {
            "qweight": "qweight",
            "qzeros": "qzeros",
            "scales": "scales",
        }
    }

    @classmethod
    def claims(cls, q: dict[str, Any]) -> bool:
        return str(q.get("quant_method") or q.get("method") or "").lower() == cls.dialect

    def __init__(self, q: dict[str, Any], hf_config: Any = None, *, name_map=None, unquantized=()):
        super().__init__(name_map, unquantized)
        bits = int(q.get("bits") or 0)
        group_size = int(q.get("group_size") or 0)
        if bits != 4 or group_size != AWQ_GROUP:
            raise NotImplementedError(
                f"AWQ configuration bits={bits}, group_size={group_size} is not supported; "
                f"only 4-bit group-{AWQ_GROUP} is supported"
            )
        if not bool(q.get("zero_point", True)):
            raise NotImplementedError("symmetric AWQ is not supported; use an asymmetric AWQ checkpoint")
        if str(q.get("version") or q.get("checkpoint_format") or "gemm").lower() != "gemm":
            raise NotImplementedError("only GEMM-packed AWQ checkpoints are supported")
        self.ignore = name_set(tuple(q.get("modules_to_not_convert") or ()))

    def scheme_for_name(self, name: str) -> QuantScheme | None:
        if self.ignore(name):
            return None
        return self.SCHEME if is_routed_expert(name) else None
