"""Unit tests for stackkernels.hub registry."""

import pytest
import stackkernels.hub as hub
from stackkernels.torch import rms_norm as torch_rms_norm


def test_hub_get_op():
    op = hub.get_op("rms_norm", backend="torch")
    assert op is torch_rms_norm

    with pytest.raises(ValueError, match="Unknown op"):
        hub.get_op("nonexistent_op")

    with pytest.raises(ValueError, match="Unknown backend"):
        hub.get_op("rms_norm", backend="tensorflow")


def test_hub_is_supported():
    assert hub.is_supported("rms_norm", dtype="float16")
    assert not hub.is_supported("nonexistent_op")

    # BF16 on sm75
    assert not hub.is_supported("rms_norm", dtype="bfloat16", compute_capability=(7, 5))
    assert hub.is_supported("rms_norm", dtype="bfloat16", compute_capability=(8, 0))

    # Attention head dim
    assert hub.is_supported("flash_attention", head_dim=64)
    assert not hub.is_supported("flash_attention", head_dim=48)
