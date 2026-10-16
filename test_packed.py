"""
Gate 1 test for quantization/packed.py (small random matrices, no model needed).

Proves that real packed storage is just a smaller way of holding the SAME
numbers: unpacking gives back exactly the weights fake quantization produced,
the layer's output is identical, and a fresh LoRA adapter changes nothing.

Run: python test_packed.py
"""

import torch
import torch.nn.functional as F

from quantization.naive_ptq import (quantize_tensor_per_channel, quantize_tensor_per_group,
                                    quantize_tensor_to_integers, quantize_tensor_to_integers_grouped)
from quantization.packed import PackedLinear, pack_integers, unpack_integers

torch.manual_seed(0)

# 1. pack / unpack is lossless
for bits in (4, 8):
    q = torch.randint(0, 2 ** bits, (6, 64))
    assert torch.equal(unpack_integers(pack_integers(q, bits), bits, 64).long(), q), f"{bits}-bit pack/unpack mismatch"
assert pack_integers(torch.randint(0, 16, (6, 64)), 4).shape == (6, 32), "4-bit must use half the bytes"
print("test 1 passed: packing and unpacking integers is lossless (4-bit halves the width)")

# 2. dequantized packed weights == fake-quantized weights, bit for bit
W = (torch.randn(48, 256) * 0.05).bfloat16()
x = torch.randn(5, 256).bfloat16()
for bits in (4, 8):
    for group in (-1, 64):
        w32 = W.float()
        if group == -1:
            q, s, z = quantize_tensor_to_integers(w32, bits)
            fake = quantize_tensor_per_channel(w32, bits).bfloat16()
        else:
            q, s, z = quantize_tensor_to_integers_grouped(w32, bits, group)
            fake = quantize_tensor_per_group(w32, bits, group).bfloat16()
        layer = PackedLinear.from_integers(q, s, z, bits, group)
        assert torch.equal(layer.dequantize(), fake), f"{bits}-bit group {group}: packed weights differ from fake-quantized ones"
        assert torch.equal(layer(x), F.linear(x, fake)), f"{bits}-bit group {group}: layer output differs"
print("test 2 passed: packed layers reproduce fake-quantized weights and outputs exactly (4/8-bit, per-row and grouped)")

# 3. a fresh LoRA adapter changes nothing (B starts at zero), and trains
layer.add_lora(rank=8, alpha=16, device="cpu")
assert torch.equal(layer(x), F.linear(x, fake)), "a fresh adapter must not change the output"
layer.lora_B.data.normal_(0, 0.01)
y = layer(x.float().requires_grad_(False).bfloat16())
y.float().pow(2).mean().backward()
assert layer.lora_A.grad is not None and layer.lora_B.grad is not None, "adapter must receive gradients"
assert not any(b.requires_grad for b in layer.buffers()), "packed weights must stay frozen"
print("test 3 passed: fresh adapter is a no-op, adapters get gradients, packed weights stay frozen")

# 4. the whole point: size
rows, cols = 4096, 4096
fp16_bytes = rows * cols * 2
packed = PackedLinear(cols, rows, 4, -1)
packed_bytes = sum(b.numel() * b.element_size() for b in packed.buffers())
print(f"\n4096x4096 layer: 16-bit {fp16_bytes / 1e6:.1f} MB -> packed 4-bit {packed_bytes / 1e6:.1f} MB ({fp16_bytes / packed_bytes:.1f}x smaller)")
print("All packed-storage tests passed.")
