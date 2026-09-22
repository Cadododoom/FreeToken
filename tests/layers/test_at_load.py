from freetoken.layers.quantization.at_load import AT_LOAD_FP8, LoadTimeFp8Config
from freetoken.layers.quantization.configs.base import NoQuantConfig


def test_glm_mla_kv_b_stays_bf16_for_absorption():
    cfg = LoadTimeFp8Config(NoQuantConfig())
    assert cfg.scheme_for("model.layers.3.self_attn.kv_b_proj") is None
    assert cfg.scheme_for("model.layers.3.self_attn.o_proj") is AT_LOAD_FP8
