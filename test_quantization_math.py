"""
Standalone test of the scale/zero-point quantization math, using plain
Python and small hand-picked numbers we can verify by hand. This has
nothing to do with the actual model yet, it's just proving the core
math is correct before we trust it on real weights.
"""

def compute_scale_and_zero_point(min_val, max_val, num_bits=8):
    """
    Computes the scale and zero-point for asymmetric quantization.

    Asymmetric means we don't assume the data is centered around 0,
    we map the ACTUAL min and max of the data onto the integer range.
    This matters because most weight distributions aren't perfectly
    symmetric, using their real min/max makes better use of the
    available integer buckets than assuming symmetry.

    For 8 bits, our integer range is 0 to 255 (256 total values).
    """
    quant_min = 0
    quant_max = (2 ** num_bits) - 1  # 255 for 8-bit

    # Scale: how much real-number range each integer step covers.
    scale = (max_val - min_val) / (quant_max - quant_min)

    # Zero-point: which integer represents real value 0.0.
    # Derived by asking "if 0.0 mapped through our scale, which integer
    # would it land on relative to min_val?"
    zero_point = quant_min - (min_val / scale)
    zero_point = round(zero_point)

    # Clamp zero_point into the valid integer range, in case of edge cases
    # (e.g. min_val and max_val both being on the same side of 0).
    zero_point = max(quant_min, min(quant_max, zero_point))

    return scale, zero_point


def quantize(value, scale, zero_point, num_bits=8):
    """Maps a real float value onto an integer in the quantized range."""
    quant_min = 0
    quant_max = (2 ** num_bits) - 1

    quantized = round(value / scale) + zero_point
    quantized = max(quant_min, min(quant_max, quantized))  # clamp to valid range
    return quantized


def dequantize(quantized_value, scale, zero_point):
    """Maps a quantized integer back to an approximate float value."""
    return (quantized_value - zero_point) * scale


# --- Verification with hand-checkable numbers ---

# Simple case: weights ranging from -2.0 to 2.0
min_val, max_val = -2.0, 2.0
scale, zero_point = compute_scale_and_zero_point(min_val, max_val)
print(f"scale={scale:.6f}, zero_point={zero_point}")

# Expected scale: (2.0 - (-2.0)) / 255 = 4.0 / 255 ≈ 0.015686
expected_scale = 4.0 / 255
assert abs(scale - expected_scale) < 1e-6, f"scale mismatch: {scale} vs {expected_scale}"

# min_val (-2.0) should quantize to close to 0 (the bottom of our integer range)
q_min = quantize(min_val, scale, zero_point)
print(f"quantize(min_val={min_val}) = {q_min} (expect close to 0)")
assert q_min == 0

# max_val (2.0) should quantize to close to 255 (the top of our integer range)
q_max = quantize(max_val, scale, zero_point)
print(f"quantize(max_val={max_val}) = {q_max} (expect close to 255)")
assert q_max == 255

# 0.0 should dequantize back to very close to 0.0
q_zero = quantize(0.0, scale, zero_point)
dq_zero = dequantize(q_zero, scale, zero_point)
print(f"quantize(0.0) -> {q_zero} -> dequantize -> {dq_zero:.6f} (expect close to 0.0)")
assert abs(dq_zero) < 0.02  # small rounding error is expected and fine

# Round-trip test: quantize then dequantize a specific value, error should
# be small (bounded by half the scale, the maximum possible rounding error)
test_value = 0.73
q = quantize(test_value, scale, zero_point)
dq = dequantize(q, scale, zero_point)
error = abs(test_value - dq)
print(f"round-trip: {test_value} -> quantize -> {q} -> dequantize -> {dq:.6f}, error={error:.6f}")
assert error <= scale, f"round-trip error {error} exceeds one scale step, something is wrong"

print("\nAll quantization math checks passed.")