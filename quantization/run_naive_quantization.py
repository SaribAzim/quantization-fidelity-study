"""
Phase 2 - Runner: load base model, apply naive quantization, evaluate,
save results.

This is the script you actually run for Phase 2. It ties together three
things we already built separately:
  - load_model.py: loads Llama 3.1 8B Instruct in BF16
  - quantization/naive_ptq.py: applies our own from-scratch quantization
  - evaluate.py: measures perplexity, latency, and memory footprint

Run this from the project ROOT (not from inside quantization/), so the
import paths below resolve correctly:
    python quantization/run_naive_quantization.py
"""

import json
import os
import sys

import torch

# Since this script lives inside quantization/ but needs to import
# load_model.py and evaluate.py from the project root, we add the root
# folder to Python's search path. We INSERT it at position 0 (front of
# the list), not append it to the end, this matters: there's a real,
# separate installed package also called "evaluate" (Hugging Face's own
# evaluation library, pulled in as a dependency of lm-eval). If we
# appended our root folder instead, Python would find that installed
# package first and never reach our own evaluate.py, exactly the
# ImportError this fix resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_model import load_base_model
from evaluate import evaluate_perplexity, evaluate_latency, evaluate_memory_footprint
from quantization.naive_ptq import apply_naive_quantization


if __name__ == "__main__":
    # Change this to run the same pipeline at a different bit width.
    # We already have 8-bit results saved (results/naive_quant_8bit.json).
    # 8-bit is known to be nearly lossless, which is exactly what we saw,
    # perplexity barely moved. The more revealing test is 4-bit, where
    # naive (uncalibrated) rounding starts to genuinely hurt accuracy,
    # this sets up the real comparison against GPTQ in Phase 3.
    NUM_BITS = 8

    model, tokenizer = load_base_model()

    print(f"\nApplying naive {NUM_BITS}-bit quantization to all Linear layers...")
    model = apply_naive_quantization(model, num_bits=NUM_BITS)

    # Save the quantized weights to disk BEFORE evaluation. Two reasons:
    # 1. lm-evaluation-harness (used for MMLU) loads a model from a path or
    #    repo ID, not from an in-memory Python object, so it needs this
    #    saved copy to run against.
    # 2. Saving first means that even if evaluation crashes afterward, we
    #    still have the quantized weights on disk and don't have to redo
    #    the (slow) loading and quantization steps from scratch.
    save_path = f"quantized_models/naive_{NUM_BITS}bit"
    print(f"\nSaving quantized model to {save_path} (for MMLU evaluation via lm-eval-harness)...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Save complete.")

    print("\nEvaluating quantized model...")
    ppl = evaluate_perplexity(model, tokenizer)
    print(f"Naive-quantized perplexity on WikiText-2: {ppl:.4f}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    tokens_per_second = evaluate_latency(model, tokenizer)
    print(f"Naive-quantized latency: {tokens_per_second:.2f} tokens/sec")

    memory_stats = evaluate_memory_footprint(model)
    print(f"Naive-quantized model size on disk: {memory_stats['model_size_gb']:.2f} GB")
    print(f"Peak VRAM during generation: {memory_stats['peak_vram_gb']:.2f} GB")

    # Note on model_size_gb here: our fake-quantization approach dequantizes
    # back to BF16, so the ACTUAL tensors in memory are still 2 bytes per
    # parameter, evaluate_memory_footprint will report the same ~15GB as
    # the base model. This is expected and worth understanding: we are
    # measuring FIDELITY loss in this phase, not real memory savings, real
    # memory savings come in Phase 3 when we compare against a properly
    # packed low-bit implementation (GPTQ/bitsandbytes) that actually
    # stores weights in reduced precision, not just simulates the rounding.

    results = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "method": f"naive_ptq_{NUM_BITS}bit",
        "perplexity": ppl,
        "tokens_per_second": tokens_per_second,
        "model_size_gb": memory_stats["model_size_gb"],
        "peak_vram_gb": memory_stats["peak_vram_gb"],
    }

    os.makedirs("results", exist_ok=True)
    results_path = f"results/naive_quant_{NUM_BITS}bit.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {results_path}")