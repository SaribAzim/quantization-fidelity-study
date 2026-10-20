"""
Exports real weights for the website's hero and its rounding explainer:
web/public/data/weights.json.

The weights come from model.layers.1.mlp.down_proj, the table the sensitivity
test found most fragile. Rounding is done on the FULL rows (all 14,336
weights, exactly as the real models were built) and only then sliced, so what
the site shows is what the models really store.

Run from the project root (CPU only):   python deployment/export_hero_weights.py
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantization.naive_ptq import (quantize_tensor_per_channel, quantize_tensor_per_group,
                                    quantize_tensor_to_integers, quantize_tensor_to_integers_grouped)
from quantization.quantize_checkpoint import find_cached_snapshot
from quantization.run_gptq import CheckpointReader

TENSOR = "model.layers.1.mlp.down_proj.weight"
ROWS, COLS = 24, 128          # hero grid
EXPLAIN_COLS = 40             # weights shown on the rounding ruler
OUT = "web/public/data/weights.json"


def r(t, d=5):
    return [round(float(v), d) for v in t.flatten().tolist()]


def main():
    weight = CheckpointReader(find_cached_snapshot()).get(TENSOR).float()   # (4096, 14336)
    block = weight[:ROWS]
    grid = {
        "original": r(block[:, :COLS]),
        "bits8": r(quantize_tensor_per_channel(block, 8)[:, :COLS]),
        "bits4": r(quantize_tensor_per_channel(block, 4)[:, :COLS]),
        "bits4Groups": r(quantize_tensor_per_group(block, 4, 128)[:, :COLS]),
    }

    # Explainer row: the row (among the first 512) where one extreme weight stretches the ruler most,
    # which is exactly why groups of 128 help.
    candidates = weight[:512]
    stretch = candidates.abs().max(dim=1).values / candidates[:, :128].abs().max(dim=1).values
    row_index = int(stretch.argmax())
    row = weight[row_index:row_index + 1]
    q, s, z = quantize_tensor_to_integers(row, 4)
    qg, sg, zg = quantize_tensor_to_integers_grouped(row, 4, 128)
    levels_row = [round(float((k - z[0, 0]) * s[0, 0]), 6) for k in range(16)]
    levels_group = [round(float((k - zg[0, 0]) * sg[0, 0]), 6) for k in range(16)]
    per_row, groups = quantize_tensor_per_channel(row, 4), quantize_tensor_per_group(row, 4, 128)
    bits8 = quantize_tensor_per_channel(row, 8)
    q8, s8, z8 = quantize_tensor_to_integers(row, 8)
    explain = {
        "bits8": r(bits8[0, :EXPLAIN_COLS], 6),
        "levels8": [round(float((k - z8[0, 0]) * s8[0, 0]), 6) for k in range(256)],
        "row": row_index,
        "original": r(row[0, :EXPLAIN_COLS], 6),
        "perRow": r(per_row[0, :EXPLAIN_COLS], 6),
        "groups": r(groups[0, :EXPLAIN_COLS], 6),
        "levelsPerRow": levels_row,
        "levelsGroup": levels_group,
        "rowMin": round(float(row.min()), 6), "rowMax": round(float(row.max()), 6),
        "groupMin": round(float(row[0, :128].min()), 6), "groupMax": round(float(row[0, :128].max()), 6),
        "meanErrorPerRow": float((row - per_row).abs().mean()),
        "meanErrorGroups": float((row - groups).abs().mean()),
        "meanError8bit": float((row - quantize_tensor_per_channel(row, 8)).abs().mean()),
    }
    out = {"tensor": TENSOR, "rows": ROWS, "cols": COLS, "grid": grid, "explain": explain,
           "distinctValues": {k: len(set(v[:COLS])) for k, v in grid.items()}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), separators=(",", ":"))
    print(f"Wrote {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB); explainer row {row_index}, "
          f"ruler range per row {explain['rowMin']:.4f}..{explain['rowMax']:.4f} vs group {explain['groupMin']:.4f}..{explain['groupMax']:.4f}; "
          f"mean error per row {explain['meanErrorPerRow']:.5f}, groups {explain['meanErrorGroups']:.5f}, 8 bit {explain['meanError8bit']:.6f}")
    print("distinct values in the first grid row:", out["distinctValues"])


if __name__ == "__main__":
    main()
