"""
Phase 2 - Naive quantization applied to the CHECKPOINT FILES, not a live model.

WHY THIS EXISTS (the meta-tensor bug):
Our first attempt loaded the model with device_map="auto" and quantized it
in memory. But the model does not fit in 8GB VRAM + 16GB RAM, so accelerate
parks some layers on disk as empty "meta" tensors (shape and dtype, but no
data). Quantizing a meta tensor silently does nothing, so only 70 of 225
Linear layers were really quantized. The old results were partial.

THE FIX: skip the live model completely. The weights already sit on disk in
.safetensors shard files. We open one shard at a time, quantize the tensors
we care about, and write a new shard. Memory use stays around the size of
one shard (~5GB) no matter how small the GPU is, and no layer can be
silently skipped because we count every tensor we touch.

WHAT GETS QUANTIZED (same set as GPTQ will use later, so results compare):
the 7 Linear weights in each of the 32 decoder layers (q,k,v,o,gate,up,down)
= 224 tensors. Embeddings, lm_head and norm weights stay in BF16.

The output is "fake quantized": values are rounded to N-bit precision and
stored back as BF16. It loads with a plain from_pretrained call and measures
FIDELITY loss in isolation. Real memory savings come from packed GPTQ models.

Run from the project root:
    python quantization/quantize_checkpoint.py --bits 8
    python quantization/quantize_checkpoint.py --bits 4
"""

import argparse
import os
import re
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantization.naive_ptq import quantize_tensor_per_channel

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# Matches e.g. "model.layers.12.self_attn.q_proj.weight" and
# "model.layers.3.mlp.down_proj.weight", and nothing else.
QUANTIZABLE_PATTERN = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)

# Small files that make the output folder a complete, loadable model.
SIDECAR_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def is_quantizable(tensor_name):
    """True if this checkpoint tensor is one of the decoder Linear weights."""
    return QUANTIZABLE_PATTERN.match(tensor_name) is not None


def quantize_shard(src_path, dst_path, num_bits, device):
    """
    Reads one .safetensors shard, quantizes the Linear weights in it, and
    writes the result. Tensors are handled one at a time so only the OUTPUT
    dict (about one shard) is held in memory.

    Returns (number quantized, number copied unchanged).
    """
    output_tensors = {}
    num_quantized = 0
    num_copied = 0

    with safe_open(src_path, framework="pt", device="cpu") as shard:
        for name in shard.keys():
            tensor = shard.get_tensor(name)

            if is_quantizable(name):
                original_dtype = tensor.dtype
                # Math in float32 for stability, same as naive_ptq.py.
                weight_fp32 = tensor.to(device=device, dtype=torch.float32)
                dequantized = quantize_tensor_per_channel(weight_fp32, num_bits=num_bits)
                output_tensors[name] = dequantized.to(original_dtype).cpu().contiguous()
                num_quantized += 1
            else:
                output_tensors[name] = tensor
                num_copied += 1

    save_file(output_tensors, dst_path, metadata={"format": "pt"})
    return num_quantized, num_copied


def quantize_checkpoint(src_dir, out_dir, num_bits, device):
    """Quantizes every shard in src_dir and writes a complete model folder."""
    os.makedirs(out_dir, exist_ok=True)

    shard_files = sorted(f for f in os.listdir(src_dir) if f.endswith(".safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors shards found in {src_dir}")

    total_quantized = 0
    total_copied = 0
    start = time.time()

    for shard_file in shard_files:
        t0 = time.time()
        print(f"Processing {shard_file} ...", flush=True)
        n_q, n_c = quantize_shard(
            os.path.join(src_dir, shard_file),
            os.path.join(out_dir, shard_file),
            num_bits,
            device,
        )
        total_quantized += n_q
        total_copied += n_c
        print(f"  quantized {n_q}, copied {n_c} ({time.time() - t0:.0f}s)", flush=True)

    for side_file in SIDECAR_FILES:
        src = os.path.join(src_dir, side_file)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(out_dir, side_file))

    print(
        f"\nDone in {time.time() - start:.0f}s. "
        f"Quantized {total_quantized} tensors to {num_bits}-bit, copied {total_copied} unchanged."
    )
    return total_quantized, total_copied


def find_cached_snapshot():
    """Locates the already-downloaded model files in the Hugging Face cache."""
    from huggingface_hub import try_to_load_from_cache

    # snapshot_download(local_files_only=True) refuses to return a cache that
    # lacks files we never use (README, original/params.json). Instead we ask
    # the cache where config.json lives; the shards sit in the same folder.
    config_path = try_to_load_from_cache(MODEL_NAME, "config.json")
    if not isinstance(config_path, str):
        raise FileNotFoundError(f"{MODEL_NAME} is not in the local Hugging Face cache")
    return os.path.dirname(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bits", type=int, required=True, choices=[2, 3, 4, 5, 6, 8])
    parser.add_argument("--src", default=None, help="folder with the original .safetensors shards")
    parser.add_argument("--out", default=None, help="output folder")
    args = parser.parse_args()

    src_dir = args.src or find_cached_snapshot()
    out_dir = args.out or f"quantized_models/naive_{args.bits}bit"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Source: {src_dir}\nOutput: {out_dir}\nDevice: {device}\n")
    n_quantized, n_copied = quantize_checkpoint(src_dir, out_dir, args.bits, device)

    # Llama 3.1 8B: 32 layers x 7 Linear weights = 224. Fail loudly if not,
    # this is exactly the silent-skip failure the old approach had.
    expected = 32 * 7
    if n_quantized != expected:
        raise SystemExit(f"ERROR: expected {expected} quantized tensors, got {n_quantized}")
    print(f"Check passed: all {expected} decoder Linear weights were quantized.")
