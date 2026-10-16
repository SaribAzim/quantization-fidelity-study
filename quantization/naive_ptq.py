"""
Phase 2 - Naive Post-Training Quantization (PTQ), applied directly to a
loaded model's weights.

The core math here (per-channel scale/zero-point, quantize, dequantize)
is exactly what we verified with plain numbers and small test tensors in
test_quantization_math.py and test_tensor_quantization_numpy.py. This file
is what actually applies that verified logic to a real 8B parameter model.

WHY WE ONLY TOUCH nn.Linear LAYERS:
A transformer model like Llama is built almost entirely out of Linear
layers (the attention projections, the feedforward layers). These are
where the vast majority of the model's parameters live, and where
quantization has the biggest size/speed impact. We leave things like
LayerNorm weights and embeddings alone for this naive first pass, they're
a small fraction of total parameters, and quantizing them carelessly can
cause disproportionate damage relative to the size savings gained.
"""

import torch
import torch.nn as nn


def quantize_tensor_per_channel(weight, num_bits=8):
    """
    Quantizes a 2D weight tensor with a separate scale and zero-point
    PER OUTPUT CHANNEL (per row), then immediately dequantizes it back to
    float. This is "fake quantization": the returned tensor has the same
    shape and dtype as the input, but now carries the rounding error real
    quantization would introduce.

    See test_tensor_quantization_numpy.py for the verified version of this
    exact logic, this is the same math, using torch instead of numpy so it
    works directly on the model's actual weight tensors.

    The rounding itself lives in quantize_tensor_to_integers below (which
    also keeps the small integers, for real packed storage); this function
    just converts those integers back to floats.
    """
    quantized, scales, zero_points = quantize_tensor_to_integers(weight, num_bits)
    return (quantized - zero_points) * scales


def quantize_tensor_to_integers(weight, num_bits=8):
    """
    The same per-row rounding as above, but returns what REAL quantization
    would store instead of converting back to floats:
        quantized    the small integers (0 .. 2**num_bits - 1), as floats
        scales       one per row, shape (rows, 1)
        zero_points  one per row, shape (rows, 1)
    Dequantizing is (quantized - zero_points) * scales.
    """
    quant_min = 0
    quant_max = (2 ** num_bits) - 1

    min_vals = weight.min(dim=1, keepdim=True).values
    max_vals = weight.max(dim=1, keepdim=True).values

    scales = (max_vals - min_vals) / (quant_max - quant_min)
    scales = torch.clamp(scales, min=1e-8)  # avoid divide-by-zero on constant rows

    zero_points = quant_min - torch.round(min_vals / scales)
    zero_points = torch.clamp(zero_points, quant_min, quant_max)

    quantized = torch.round(weight / scales) + zero_points
    quantized = torch.clamp(quantized, quant_min, quant_max)

    return quantized, scales, zero_points


def quantize_tensor_to_integers_grouped(weight, num_bits=8, group_size=128):
    """
    Grouped version of quantize_tensor_to_integers (same reshape trick as
    quantize_tensor_per_group). Returns quantized with shape (rows, cols) and
    scales / zero_points with shape (rows, cols // group_size).
    """
    rows, cols = weight.shape
    assert cols % group_size == 0, f"{cols} columns not divisible by group size {group_size}"
    quantized, scales, zero_points = quantize_tensor_to_integers(
        weight.reshape(rows * cols // group_size, group_size), num_bits)
    return (quantized.reshape(rows, cols), scales.reshape(rows, cols // group_size),
            zero_points.reshape(rows, cols // group_size))


def quantize_tensor_per_group(weight, num_bits=8, group_size=128):
    """
    Same rounding as quantize_tensor_per_channel, but each scale/zero-point
    covers only `group_size` neighbouring weights of a row instead of the
    whole row. Smaller groups follow the weights' local range more closely,
    so there is less rounding error, at the cost of storing more scales.

    Trick: a row-major matrix of shape (rows, cols) reshaped to
    (rows * cols / group_size, group_size) puts each group on its own row,
    so the verified per-channel function can be reused unchanged.
    """
    rows, cols = weight.shape
    assert cols % group_size == 0, f"{cols} columns not divisible by group size {group_size}"
    grouped = weight.reshape(rows * cols // group_size, group_size)
    return quantize_tensor_per_channel(grouped, num_bits=num_bits).reshape(rows, cols)


def apply_naive_quantization(model, num_bits=8):
    """
    Walks through every nn.Linear layer in the model and replaces its
    weight with a fake-quantized version, in place.

    Why in place, and why .data: modifying model.named_modules() lets us
    reach every layer regardless of how deeply nested it is inside the
    model's architecture, without needing to know Llama's exact structure
    ahead of time. We modify .data directly (not the weight parameter
    itself) because we're replacing VALUES, not creating a new trainable
    parameter, this avoids interfering with the model's parameter
    bookkeeping (gradients, optimizer state) that we're not using anyway
    since we're doing inference-only evaluation, not training.

    IMPORTANT DIAGNOSTIC, added after noticing a real risk: when a model is
    loaded with device_map="auto" and disk offloading, some layers may
    still be "meta" tensors at this point, placeholders with a shape and
    dtype but NO actual data, waiting to be materialized just before use.
    If we tried to quantize a meta tensor, the operation would silently
    produce another meta tensor (no real computation, no error either),
    meaning that layer's weights would NOT actually get quantized despite
    the code appearing to run successfully. We explicitly check for this
    and report it, rather than silently trusting that everything worked.

    Args:
        model: an already-loaded HuggingFace causal LM
        num_bits: bit width to quantize each Linear layer's weights to

    Returns:
        The same model object, modified in place (returned for convenience,
        chaining, so callers can write `model = apply_naive_quantization(model)`
        even though modification happens in place either way).
    """
    num_layers_quantized = 0
    num_layers_skipped_meta = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module.weight.data.is_meta:
                # This layer's weight has no real data right now, quantizing
                # it would be a silent no-op. We skip it explicitly and
                # count it, rather than pretending we quantized it.
                num_layers_skipped_meta += 1
                continue

            original_dtype = module.weight.data.dtype

            # Quantization math is done in float32 for numerical stability
            # during the min/max and division steps, then cast back to the
            # model's original precision (BF16) afterward, so the model's
            # overall dtype stays consistent.
            weight_fp32 = module.weight.data.to(torch.float32)
            dequantized = quantize_tensor_per_channel(weight_fp32, num_bits=num_bits)
            module.weight.data = dequantized.to(original_dtype)

            num_layers_quantized += 1

    print(f"Applied {num_bits}-bit naive quantization to {num_layers_quantized} Linear layers.")
    if num_layers_skipped_meta > 0:
        print(
            f"WARNING: {num_layers_skipped_meta} Linear layers were on the meta device "
            f"(no real data present yet) and were SKIPPED, not quantized. This means "
            f"the model is only PARTIALLY quantized. See the note in this file's "
            f"docstring for why this happens and how to fix it."
        )

    return model