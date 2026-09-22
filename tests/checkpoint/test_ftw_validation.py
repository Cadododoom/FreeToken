import json

import pytest
import torch

from freetoken.checkpoint.ftw import FTWFormatError, FTWReader, FTWWriter


def _index(tmp_path):
    writer = FTWWriter(str(tmp_path))
    writer.add_tensor("model.weight", torch.zeros(4, dtype=torch.bfloat16))
    return writer.finalize({})


def _rewrite(tmp_path, index):
    (tmp_path / "freetoken_weight.json").write_text(json.dumps(index))


def test_ftw_reader_rejects_shard_size_mismatch(tmp_path):
    index = _index(tmp_path)
    index["shards"][0]["nbytes"] += 4096
    _rewrite(tmp_path, index)
    with pytest.raises(FTWFormatError, match="indexed size"):
        FTWReader(str(tmp_path))


def test_ftw_reader_rejects_shard_path_traversal(tmp_path):
    index = _index(tmp_path)
    index["shards"][0]["file"] = "../outside.ftw"
    _rewrite(tmp_path, index)
    with pytest.raises(FTWFormatError, match="safe basename"):
        FTWReader(str(tmp_path))
