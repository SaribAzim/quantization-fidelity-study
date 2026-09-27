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

    dequantized_weight = (quantized - zero_points) * scales
    return dequantized_weight


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

    Args:
        model: an already-loaded HuggingFace causal LM
        num_bits: bit width to quantize each Linear layer's weights to

    Returns:
        The same model object, modified in place (returned for convenience,
        chaining, so callers can write `model = apply_naive_quantization(model)`
        even though modification happens in place either way).
    """
    num_layers_quantized = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
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
    return model