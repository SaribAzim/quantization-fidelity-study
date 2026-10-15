"""
Tests for quantization/gptq.py on small random matrices (no model needed).

Test 1 proves GPTQ reduces to naive rounding when there is nothing to
compensate (H = identity: inputs uncorrelated, all equally important). That
confirms the grid/rounding is the same code path as naive.

Test 2 proves the point of GPTQ: with CORRELATED inputs, the layer's output
error is smaller than with naive rounding.

Run: python test_gptq.py
"""

import torch

from quantization.gptq import HessianAccumulator, gptq_quantize
from quantization.naive_ptq import quantize_tensor_per_channel, quantize_tensor_per_group

torch.manual_seed(0)
W = torch.randn(48, 256)

# --- Test 1: identity Hessian -> exactly naive rounding -------------------
eye = torch.eye(256)
for bits in (8, 4):
    assert torch.allclose(gptq_quantize(W, eye, bits), quantize_tensor_per_channel(W, bits), atol=1e-6), \
        f"{bits}-bit per-channel: GPTQ with H=I must equal naive"
    assert torch.allclose(gptq_quantize(W, eye, bits, group_size=64), quantize_tensor_per_group(W, bits, 64), atol=1e-6), \
        f"{bits}-bit group-64: GPTQ with H=I must equal naive"
print("test 1 passed: with H = identity, GPTQ == naive rounding (per-channel and grouped)")

# --- Test 2: correlated inputs -> GPTQ beats naive on OUTPUT error ---------
mixing = torch.randn(256, 256) * 0.15 + torch.eye(256)  # makes input features correlated
X = torch.randn(4096, 256) @ mixing                      # 4096 fake input vectors
acc = HessianAccumulator(256, "cpu")
for chunk in X.split(512):
    acc.add_batch(chunk)
assert acc.num_seen == 4096
assert torch.allclose(acc.H, 2 * X.T @ X / 4096, rtol=1e-3, atol=1e-3), "running Hessian must equal the direct formula"


def output_error(Q):
    """How far the layer's OUTPUTS move (what actually matters), not the weights."""
    return ((W - Q) @ X.T).pow(2).mean().item()


for bits, group in ((4, -1), (4, 64), (3, -1)):
    naive = quantize_tensor_per_channel(W, bits) if group == -1 else quantize_tensor_per_group(W, bits, group)
    gptq = gptq_quantize(W, acc.H, bits, group_size=group)
    ratio = output_error(gptq) / output_error(naive)
    print(f"  {bits}-bit, group {group}: output error GPTQ/naive = {ratio:.3f}")
    assert ratio < 0.95, "GPTQ should clearly reduce output error with correlated inputs"
    # Every row still uses at most 2^bits distinct values per group.
    rows_to_check = gptq if group == -1 else gptq.reshape(-1, group)
    assert max(r.unique().numel() for r in rows_to_check) <= 2 ** bits

print("test 2 passed: GPTQ lowers layer output error vs naive when inputs are correlated")
print("\nAll GPTQ tests passed.")
