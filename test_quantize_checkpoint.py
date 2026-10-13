"""
Synthetic test for quantization/quantize_checkpoint.py. No model download
needed: we build a tiny fake checkpoint with the real tensor names, run the
shard quantizer on it, and check the output is what we claim.

Run: python test_quantize_checkpoint.py
"""

import os
import tempfile

import torch
from safetensors.torch import load_file, save_file

from quantization.naive_ptq import quantize_tensor_per_channel
from quantization.quantize_checkpoint import is_quantizable, quantize_shard

# 1. Name matching: exactly the 7 decoder Linear weights per layer.
assert is_quantizable("model.layers.0.self_attn.q_proj.weight")
assert is_quantizable("model.layers.31.mlp.down_proj.weight")
assert not is_quantizable("model.embed_tokens.weight")
assert not is_quantizable("lm_head.weight")
assert not is_quantizable("model.norm.weight")
assert not is_quantizable("model.layers.0.input_layernorm.weight")
assert not is_quantizable("model.layers.0.self_attn.q_proj.bias")

# 2. Shard quantization on a small fake shard.
torch.manual_seed(0)
fake = {
    "model.layers.0.self_attn.q_proj.weight": torch.randn(16, 64).bfloat16(),
    "model.layers.0.mlp.up_proj.weight": torch.randn(32, 64).bfloat16(),
    "model.layers.0.input_layernorm.weight": torch.ones(64).bfloat16(),
    "model.embed_tokens.weight": torch.randn(100, 64).bfloat16(),
}

# ignore_cleanup_errors: on Windows, safetensors keeps the output file
# memory-mapped until garbage collection, so deleting it can fail harmlessly.
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    src, dst = os.path.join(tmp, "in.safetensors"), os.path.join(tmp, "out.safetensors")
    save_file(fake, src)

    for bits in (8, 4):
        n_q, n_c = quantize_shard(src, dst, bits, device="cpu")
        assert (n_q, n_c) == (2, 2), (n_q, n_c)
        out = load_file(dst)

        assert set(out) == set(fake), "tensor names must be unchanged"
        for name in fake:
            assert out[name].dtype == torch.bfloat16 and out[name].shape == fake[name].shape

        # Untouched tensors are bit-identical.
        for name in ("model.layers.0.input_layernorm.weight", "model.embed_tokens.weight"):
            assert torch.equal(out[name], fake[name]), f"{name} should be copied unchanged"

        # Quantized tensors match the verified naive_ptq math exactly...
        for name in ("model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.up_proj.weight"):
            expected = quantize_tensor_per_channel(fake[name].float(), bits).bfloat16()
            assert torch.equal(out[name], expected), f"{name} differs from naive_ptq output"

            # ...and each row uses at most 2^bits distinct values (that is
            # what "N-bit per channel" means). Check in float32: BF16 would
            # merge some of them, so this bound is never violated by it.
            for row in out[name].float():
                assert row.unique().numel() <= 2 ** bits, "row has too many distinct values"

        print(f"{bits}-bit shard test passed")

print("\nAll quantize_checkpoint tests passed.")
