"""Tests executing code examples from README.md and docs/ on CPU with reference backend,
plus a relative link checker verifying that all markdown links point to existing files.
"""

import os
import re
from pathlib import Path
import pytest
import torch

import stackkernels
from stackkernels.torch import (
    CrossEntropyLoss,
    RMSNorm,
    RoPE,
    flash_attention,
    rms_norm,
    swiglu,
)
from stackkernels._opt.pick import pick
from stackkernels._opt.tune import tune


@pytest.fixture(autouse=True)
def force_reference_backend():
    """Ensure CPU-safe execution using reference backend for docs example testing."""
    prev = stackkernels.get_backend()
    stackkernels.options(backend="reference")
    yield
    stackkernels.options(backend=prev)


def test_readme_quickstart_torch():
    """Test the PyTorch functional and module quickstart examples from README.md."""
    device = torch.device("cpu")

    # 1. Functional RMSNorm
    x = torch.randn(2, 64, 128, device=device, requires_grad=True)
    weight = torch.ones(128, device=device)
    y = rms_norm(x, weight)
    assert y.shape == (2, 64, 128)
    y.sum().backward()
    assert x.grad is not None

    # 2. Functional SwiGLU
    gate = torch.randn(2, 64, 256, device=device)
    up = torch.randn(2, 64, 256, device=device)
    act = swiglu(gate, up)
    assert act.shape == (2, 64, 256)

    # 3. Functional FlashAttention
    q = torch.randn(2, 4, 64, 32, device=device)
    k = torch.randn(2, 2, 64, 32, device=device)  # GQA
    v = torch.randn(2, 2, 64, 32, device=device)
    attn_out = flash_attention(q, k, v, is_causal=True)
    assert attn_out.shape == (2, 4, 64, 32)

    # 4. Modules
    norm = RMSNorm(dim=128)
    norm_out = norm(x)
    assert norm_out.shape == (2, 64, 128)

    rope = RoPE(dim=32, max_seq_len=512)
    q_rot, k_rot = rope(q, k)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape

    criterion = CrossEntropyLoss(ignore_index=-100)
    logits = torch.randn(128, 512, device=device)
    targets = torch.randint(0, 512, (128,), device=device)
    loss = criterion(logits, targets)
    assert loss.dim() == 0


def test_readme_quickstart_optimization():
    """Test optimization pick explain and tune examples from README.md on CPU."""
    cfg, explanation = pick("rms_norm", shape=(64, 128), dtype="float32", explain=True)
    assert isinstance(cfg, dict)
    assert isinstance(explanation, str)
    assert "rms_norm" in explanation

    # tune on CPU records baseline configuration without GPU benchmarking
    best_config = tune("rms_norm", shape=(64, 128), dtype="float32", write_to_disk=False)
    assert isinstance(best_config, dict)
    assert "BLOCK_SIZE" in best_config


def test_readme_quickstart_jax():
    """Test JAX functional quickstart from README.md."""
    try:
        import jax
        import jax.numpy as jnp
        from stackkernels.jax import rms_norm as jax_rms_norm

        x = jnp.ones((2, 32, 64), dtype=jnp.float32)
        w = jnp.ones((64,), dtype=jnp.float32)

        grad_x = jax.grad(lambda x_in: jnp.sum(jax_rms_norm(x_in, w)))(x)
        assert grad_x.shape == (2, 32, 64)
    except ImportError:
        pytest.skip("JAX not installed in test environment")


def test_getting_started_example():
    """Test first call example from docs/getting_started.md."""
    device = torch.device("cpu")
    x = torch.randn(2, 64, 128, device=device, dtype=torch.float32)
    weight = torch.ones(128, device=device, dtype=torch.float32)
    y = rms_norm(x, weight=weight, eps=1e-6)
    assert y.shape == x.shape


def test_stackformer_integration_script():
    """Test that examples/stackformer_integration.py runs without error on CPU."""
    from examples.stackformer_integration import StackformerBlock

    dim = 64
    num_heads = 4
    hidden_dim = 128
    block = StackformerBlock(dim, num_heads, hidden_dim)

    x = torch.randn(2, 32, dim)
    out = block(x)
    assert out.shape == x.shape


def test_markdown_relative_links():
    """Verify that all relative markdown links in README.md and docs/ resolve to existing files."""
    repo_root = Path(__file__).resolve().parent.parent
    md_files = [repo_root / "README.md", repo_root / "CONTRIBUTING.md"] + list((repo_root / "docs").rglob("*.md"))

    # Regex matching markdown links [text](path)
    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    broken_links = []

    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        # Strip out code blocks to avoid testing false links in code snippets
        content_no_code = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
        content_no_inline = re.sub(r"`.*?`", "", content_no_code)

        for match in link_pattern.finditer(content_no_inline):
            link = match.group(2).strip()

            # Ignore external URLs, mailto, image links with http, and in-page anchor fragments
            if (
                link.startswith("http://")
                or link.startswith("https://")
                or link.startswith("mailto:")
                or link.startswith("#")
            ):
                continue

            # Strip query params or anchors from link
            clean_link = link.split("#")[0].split("?")[0]
            if not clean_link:
                continue

            # Resolve relative to the directory containing the markdown file
            target_path = (md_file.parent / clean_link).resolve()

            if not target_path.exists():
                broken_links.append(f"{md_file.relative_to(repo_root)} -> {link} (resolved: {target_path})")

    assert not broken_links, "Found broken relative markdown links:\n" + "\n".join(broken_links)

