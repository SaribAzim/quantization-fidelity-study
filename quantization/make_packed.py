"""
Builds a REAL packed 4-bit (or 8-bit) model folder from the original weights.

Same rounding code as the naive fake-quantized models (so the dequantized
weights are identical), but the small integers, scales and zero-points are
kept and written to disk instead of being converted back to 16-bit floats.

Run from the project root:
    python quantization/make_packed.py --bits 4                       (one ruler per row, the model used so far)
    python quantization/make_packed.py --bits 4 --group-size 128
Output: quantized_models/packed_naive_4bit  (about 5.3 GB instead of 15 GB)
"""

import argparse
import json
import os
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantization.naive_ptq import quantize_tensor_to_integers, quantize_tensor_to_integers_grouped
from quantization.packed import pack_integers
from quantization.quantize_checkpoint import SIDECAR_FILES, find_cached_snapshot, is_quantizable


def pack_shard(src_path, dst_path, bits, group_size, device):
    """Writes one packed shard. Returns the new tensor names it contains."""
    out = {}
    with safe_open(src_path, framework="pt", device="cpu") as shard:
        for name in shard.keys():
            tensor = shard.get_tensor(name)
            if is_quantizable(name):
                weight = tensor.to(device=device, dtype=torch.float32)
                if group_size == -1:
                    q, scales, zeros = quantize_tensor_to_integers(weight, bits)
                else:
                    q, scales, zeros = quantize_tensor_to_integers_grouped(weight, bits, group_size)
                stem = name[:-len(".weight")]
                out[f"{stem}.qweight"] = pack_integers(q, bits).cpu().contiguous()
                out[f"{stem}.scales"] = scales.reshape(weight.size(0), -1).cpu().contiguous()
                out[f"{stem}.zeros"] = zeros.reshape(weight.size(0), -1).cpu().contiguous()
            else:
                out[name] = tensor
    save_file(out, dst_path, metadata={"format": "pt"})
    return list(out.keys())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8])
    parser.add_argument("--group-size", type=int, default=-1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    src_dir = find_cached_snapshot()
    suffix = "" if args.group_size == -1 else f"_g{args.group_size}"
    out_dir = args.out or f"quantized_models/packed_naive_{args.bits}bit{suffix}"
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    weight_map, start = {}, time.time()
    shards = sorted(f for f in os.listdir(src_dir) if f.endswith(".safetensors"))
    for i, shard in enumerate(shards):
        new_name = f"packed-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        print(f"Packing {shard} -> {new_name}", flush=True)
        for key in pack_shard(os.path.join(src_dir, shard), os.path.join(out_dir, new_name), args.bits, args.group_size, device):
            weight_map[key] = new_name

    json.dump({"metadata": {}, "weight_map": weight_map}, open(os.path.join(out_dir, "model.safetensors.index.json"), "w"))
    json.dump({"method": "naive_rtn", "bits": args.bits, "group_size": args.group_size, "overrides": {}},
              open(os.path.join(out_dir, "packed_config.json"), "w"), indent=2)
    for side_file in SIDECAR_FILES:
        src = os.path.join(src_dir, side_file)
        if os.path.exists(src) and side_file != "model.safetensors.index.json":
            shutil.copyfile(src, os.path.join(out_dir, side_file))
    size = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir)) / 1024 ** 3
    print(f"Done in {time.time() - start:.0f}s: {out_dir} is {size:.2f} GB on disk")


if __name__ == "__main__":
    main()
