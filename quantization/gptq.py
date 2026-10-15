"""
Phase 3 - GPTQ, written from scratch (the algorithm of Frantar et al., 2022).

THE PROBLEM WITH NAIVE ROUNDING:
Naive quantization rounds every weight on its own, as if each weight were
independent. But a layer computes  output = W @ x  and the INPUTS x are
correlated: when two input features usually move together, an error in one
weight can be cancelled by a small opposite change in a neighbouring weight.
Naive rounding never uses that.

WHAT GPTQ DOES INSTEAD:
1. Feed a little real text ("calibration data") through the model and record
   what goes INTO each Linear layer. From that build the matrix
        H = 2 * average of (x x^T)        (the "Hessian")
   H[i][j] says how much input features i and j matter and how they move
   together. A large H[i][i] means "an error in column i hurts a lot".
2. Quantize the weight matrix ONE COLUMN AT A TIME. After rounding column i,
   measure the error it caused and spread a correction over all the columns
   that are NOT yet quantized, so they absorb it. The correction weights come
   from the inverse of H (this is the "optimal brain surgeon" idea).
3. Columns are processed in blocks of 128, and the correction for columns
   outside the current block is applied once per block (faster, same result).

WHAT STAYS EXACTLY THE SAME AS NAIVE: the grid of allowed values. Scale and
zero-point come from the same min/max formulas as quantize_tensor_per_channel,
so the ONLY difference between naive and GPTQ here is the error compensation.
That makes the comparison a clean test of "does calibration help?".
"""

import math

import torch


def find_scale_zero(block, num_bits):
    """
    Per-row scale and zero-point for a (rows, k) block. Identical formulas to
    quantize_tensor_per_channel in naive_ptq.py (asymmetric min/max).
    """
    quant_max = (2 ** num_bits) - 1
    min_vals = block.min(dim=1, keepdim=True).values
    max_vals = block.max(dim=1, keepdim=True).values
    scales = ((max_vals - min_vals) / quant_max).clamp(min=1e-8)
    zero_points = (0 - torch.round(min_vals / scales)).clamp(0, quant_max)
    return scales, zero_points


def quantize_column(column, scales, zero_points, num_bits):
    """Rounds one column (one weight per output row) to the grid and back."""
    quant_max = (2 ** num_bits) - 1
    q = (torch.round(column.unsqueeze(1) / scales) + zero_points).clamp(0, quant_max)
    return ((q - zero_points) * scales).squeeze(1)


class HessianAccumulator:
    """
    Builds H = 2/N * sum of (x x^T) over every input vector x that a Linear
    layer sees, without ever holding all the inputs at once: each batch is
    folded into a running average.
    """

    def __init__(self, num_cols, device):
        self.H = torch.zeros(num_cols, num_cols, dtype=torch.float32, device=device)
        self.num_seen = 0

    def add_batch(self, x):
        x = x.reshape(-1, x.shape[-1]).float()  # (tokens, cols)
        new = x.shape[0]
        self.H *= self.num_seen / (self.num_seen + new)  # shrink old average
        self.num_seen += new
        x = math.sqrt(2 / self.num_seen) * x
        self.H += x.T @ x


def gptq_quantize(weight, H, num_bits, group_size=-1, percdamp=0.01, blocksize=128):
    """
    Quantizes one weight matrix with GPTQ and returns the dequantized
    ("fake quantized") matrix in float32, same shape as `weight`.

    Args:
        weight: (rows, cols) matrix, rows = output features, cols = inputs.
        H: (cols, cols) Hessian from HessianAccumulator.
        num_bits: bit width.
        group_size: -1 for one scale per row, else one scale per `group_size`
            columns (must divide cols). Groups use blocksize = group_size so
            each group's scale is picked from weights that already contain
            the corrections from all earlier columns.
        percdamp: adds this fraction of the average diagonal of H to its
            diagonal. H can be nearly singular; this keeps inversion stable.
    """
    W = weight.clone().float()
    rows, cols = W.shape
    H = H.clone()

    # Input features that never fire (H diagonal 0) carry no information:
    # zero their weights, and give H a harmless 1 so it stays invertible.
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    diag_index = torch.arange(cols, device=W.device)
    H[diag_index, diag_index] += percdamp * torch.mean(torch.diag(H))

    # Hinv = upper Cholesky factor of H^-1. Row i of it gives, for column i,
    # how to spread its rounding error over the columns to its right.
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    Hinv = torch.linalg.cholesky(H, upper=True)
    del H

    if group_size == -1:
        scales, zero_points = find_scale_zero(W, num_bits)  # one grid per row
    else:
        assert cols % group_size == 0, f"{cols} columns not divisible by {group_size}"
        blocksize = group_size

    Q = torch.zeros_like(W)
    for start in range(0, cols, blocksize):
        end = min(start + blocksize, cols)
        block = W[:, start:end].clone()
        block_q = torch.zeros_like(block)
        block_err = torch.zeros_like(block)
        Hinv_block = Hinv[start:end, start:end]

        if group_size != -1:
            scales, zero_points = find_scale_zero(block, num_bits)

        for i in range(end - start):
            w = block[:, i]
            q = quantize_column(w, scales, zero_points, num_bits)
            block_q[:, i] = q
            err = (w - q) / Hinv_block[i, i]
            # Push the error onto the not-yet-quantized columns of this block.
            block[:, i:] -= err.unsqueeze(1) @ Hinv_block[i, i:].unsqueeze(0)
            block_err[:, i] = err

        Q[:, start:end] = block_q
        # ...and onto every column in later blocks, in one matrix product.
        W[:, end:] -= block_err @ Hinv[start:end, end:]

    return Q
