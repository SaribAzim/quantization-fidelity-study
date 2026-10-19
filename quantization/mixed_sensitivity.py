"""
Mixed precision, step 1: which parts of the model suffer most from 4-bit rounding?

Idea: start from the real packed 4-bit model. Take one group of layers, swap
them to 8-bit (which is almost lossless), and measure how much the model's
output predictions CHANGE. A group whose upgrade changes the output a lot is a
group whose 4-bit rounding was hurting a lot, so it is worth protecting.

Everything here uses WikiText VALIDATION text, never the test text or the exam,
so choosing what to protect cannot leak into the reported results.

  * per-layer: all 7 weight tables of one decoder layer at 8-bit (32 trials)
  * per-type:  one kind of table (q, k, v, o, gate, up, down) at 8-bit in four
               spread-out layers (7 trials), to see which KIND matters most

Run from the project root:   python quantization/mixed_sensitivity.py
Output: results/mixed/sensitivity.json
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import evaluate_perplexity_and_fidelity
from quantization.naive_ptq import quantize_tensor_to_integers
from quantization.packed import PackedLinear, load_packed_model
from quantization.quantize_checkpoint import find_cached_snapshot
from quantization.run_gptq import CheckpointReader

TYPES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
TYPE_TEST_LAYERS = [4, 12, 20, 28]  # four spread-out layers for the per-type trials
REFERENCE = "results/mixed/ref_all4bit_validation.pt"


def swap_to_8bit(model, reader, module_names):
    """Replaces the named layers with 8-bit versions built from the ORIGINAL weights; returns the originals."""
    saved = {}
    for name in module_names:
        weight = reader.get(name + ".weight").to("cuda", torch.float32)
        q, scales, zeros = quantize_tensor_to_integers(weight, 8)
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        saved[name] = getattr(parent, child)
        setattr(parent, child, PackedLinear.from_integers(q, scales, zeros, 8, -1))
        del weight
    return saved


def restore(model, saved):
    for name, module in saved.items():
        parent_name, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent_name), child, module)
    torch.cuda.empty_cache()


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "quantized_models/packed_naive_4bit"
    model, tokenizer = load_packed_model(folder)
    reader = CheckpointReader(find_cached_snapshot())
    os.makedirs("results/mixed", exist_ok=True)

    # The all-4-bit model's own predictions are the yardstick: KL measures how much an upgrade changes them.
    evaluate_perplexity_and_fidelity(model, tokenizer, REFERENCE, save_reference=True, split="validation")

    def trial(names):
        saved = swap_to_8bit(model, reader, names)
        stats = evaluate_perplexity_and_fidelity(model, tokenizer, REFERENCE, split="validation")
        restore(model, saved)
        return {"kl_change": stats["kl_divergence"], "top1_same": stats["top1_agreement"], "perplexity": stats["perplexity"]}

    out = {"per_layer": {}, "per_type": {}, "type_test_layers": TYPE_TEST_LAYERS}
    start = time.time()
    for layer in range(32):
        out["per_layer"][layer] = trial([f"model.layers.{layer}.{t}" for t in TYPES])
        print(f"layer {layer:2d}: output changes by KL {out['per_layer'][layer]['kl_change']:.4f} when upgraded "
              f"({time.time() - start:.0f}s)", flush=True)
    for t in TYPES:
        out["per_type"][t] = trial([f"model.layers.{l}.{t}" for l in TYPE_TEST_LAYERS])
        print(f"type {t:18s}: KL {out['per_type'][t]['kl_change']:.4f}", flush=True)
    json.dump(out, open("results/mixed/sensitivity.json", "w"), indent=1)
    print("Saved results/mixed/sensitivity.json")


if __name__ == "__main__":
    main()
