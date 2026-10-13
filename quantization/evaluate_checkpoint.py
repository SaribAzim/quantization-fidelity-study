"""
Phase 2/3 - Evaluate ONE model on every metric, loading it only once.

Loading an offloaded 8B model is the slowest step, so instead of one script
per metric we load once and run everything, saving each stage as soon as it
finishes. If the run is interrupted, running the same command again skips
the stages already stored in the results file.

Metrics (identical code for every model, that is the point of the project):
  1. perplexity            WikiText-2, first 50 samples, stride 512
  2. fidelity vs base      KL divergence + top-1 agreement on the same text
  3. latency + memory      tokens/sec, real size in memory, peak VRAM
  4. MMLU                  4 computer-science subtasks, 0-shot, chat template,
                           via lm-evaluation-harness (see below)

MMLU detail: we score all 412 questions of high_school_computer_science,
college_computer_science, computer_security and machine_learning once. The
project's original metric (first 50 questions of high_school_computer_science,
what `lm_eval --limit 50` does) is just the doc_ids < 50 of that task, so we
derive it from the same run instead of running twice.

Examples (run from the project root):
    python quantization/evaluate_checkpoint.py --model meta-llama/Llama-3.1-8B-Instruct --name baseline --method bf16 --save-reference
    python quantization/evaluate_checkpoint.py --model quantized_models/naive_8bit --name naive_quant_8bit --method naive_ptq --bits 8
"""

import argparse
import json
import math
import os
import sys

import torch

# insert(0): our evaluate.py must win over the installed 'evaluate' package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import (
    estimate_packed_size_gb,
    evaluate_latency,
    evaluate_memory_footprint,
    evaluate_perplexity_and_fidelity,
    is_fully_on_gpu,
)
from load_model import load_model_from

REFERENCE_PATH = "results/reference/base_wikitext_topk.pt"
MMLU_TASKS = [
    "mmlu_high_school_computer_science",
    "mmlu_college_computer_science",
    "mmlu_computer_security",
    "mmlu_machine_learning",
]
ORIGINAL_TASK = "mmlu_high_school_computer_science"
ORIGINAL_N = 50
# Weight streaming dominates offloaded models, so a larger batch amortizes it.
# The same value is used for every model so results stay comparable.
MMLU_BATCH_SIZE = 16


def accuracy_and_stderr(correct_flags):
    """Mean accuracy and the sample standard error lm-eval reports (n-1)."""
    n = len(correct_flags)
    acc = sum(correct_flags) / n
    stderr = math.sqrt(acc * (1 - acc) / (n - 1)) if n > 1 else float("nan")
    return acc, stderr


def run_mmlu(model, tokenizer, name):
    """Runs lm-eval on the already-loaded model; returns a results dict."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=MMLU_BATCH_SIZE)
    out = lm_eval.simple_evaluate(
        model=lm, tasks=MMLU_TASKS, num_fewshot=0, apply_chat_template=True, log_samples=True
    )

    per_question = {}  # task -> list of (doc_id, 0/1)
    for task, samples in out["samples"].items():
        per_question[task] = [(s["doc_id"], int(s["acc"])) for s in samples]

    # Keep per-question outcomes: later phases compare models question by
    # question (paired), which is far more sensitive than comparing totals.
    os.makedirs("results/samples", exist_ok=True)
    with open(f"results/samples/{name}_mmlu.json", "w") as f:
        json.dump(per_question, f)

    per_task = {}
    all_flags = []
    for task, pairs in per_question.items():
        flags = [c for _, c in pairs]
        acc, se = accuracy_and_stderr(flags)
        per_task[task] = {"accuracy": acc, "stderr": se, "n": len(flags)}
        all_flags += flags
    ext_acc, ext_se = accuracy_and_stderr(all_flags)

    original_flags = [c for d, c in per_question[ORIGINAL_TASK] if d < ORIGINAL_N]
    orig_acc, orig_se = accuracy_and_stderr(original_flags)

    return {
        "mmlu_task": "high_school_computer_science",
        "mmlu_accuracy": orig_acc,
        "mmlu_stderr": orig_se,
        "mmlu_num_samples": len(original_flags),
        "mmlu_extended_cs": {"accuracy": ext_acc, "stderr": ext_se, "n": len(all_flags), "per_task": per_task},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model folder or hub repo ID")
    parser.add_argument("--name", required=True, help="results/<name>.json")
    parser.add_argument("--method", required=True)
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=-1, help="-1 = per-channel")
    parser.add_argument("--save-reference", action="store_true", help="this IS the base model: store its predictions")
    parser.add_argument("--gpu-only", action="store_true", help="force the whole model onto GPU 0 (packed models)")
    parser.add_argument("--skip-mmlu", action="store_true")
    args = parser.parse_args()

    results_path = f"results/{args.name}.json"
    os.makedirs("results", exist_ok=True)
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}
    results.update({"model": "meta-llama/Llama-3.1-8B-Instruct", "method": args.method,
                    "bits": args.bits, "group_size": args.group_size})

    def save():
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    need_fidelity = "perplexity" not in results or ("kl_divergence" not in results and not args.save_reference)
    need_speed = "tokens_per_second" not in results
    need_mmlu = "mmlu_extended_cs" not in results and not args.skip_mmlu
    if not (need_fidelity or need_speed or need_mmlu):
        print(f"Everything already in {results_path}, nothing to do.")
        return

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model_from(args.model, device_map={"": 0} if args.gpu_only else "auto")
    results["fully_on_gpu"] = is_fully_on_gpu(model)

    if need_fidelity:
        stats = evaluate_perplexity_and_fidelity(
            model, tokenizer, REFERENCE_PATH, save_reference=args.save_reference
        )
        results.update(stats)
        print(f"perplexity={stats['perplexity']:.4f} kl={stats['kl_divergence']} top1={stats['top1_agreement']}")
        save()

    if need_speed:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        results["tokens_per_second"] = evaluate_latency(model, tokenizer)
        memory = evaluate_memory_footprint(model)
        results["model_size_gb"] = memory["model_size_gb"]
        results["peak_vram_gb"] = memory["peak_vram_gb"]
        if args.bits < 16 and args.method.startswith("naive"):
            # Fake-quantized models are stored as BF16; this is what a packed
            # file of the same weights would take.
            results["packed_size_gb_estimate"] = estimate_packed_size_gb(model, args.bits, args.group_size)
        print(f"{results['tokens_per_second']:.2f} tok/s, fully_on_gpu={results['fully_on_gpu']}")
        save()

    if need_mmlu:
        results.update(run_mmlu(model, tokenizer, args.name))
        print(f"MMLU n=50: {results['mmlu_accuracy']:.3f}  extended CS: {results['mmlu_extended_cs']['accuracy']:.3f}")
        save()

    print(f"Saved {results_path}")


if __name__ == "__main__":
    main()
