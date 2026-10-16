"""
Phase 4/5 - Run the RAG experiment for ONE model.

Three ways of answering the same 412 CS exam questions with the same model:
  none   no retrieval (must reproduce the earlier lm-eval numbers: validation gate)
  plain  search the library with the question and its options, add the top passages to the prompt
  hyde   have the model write a made-up answer passage, search with THAT, add the top passages

Everything is saved to results/rag/<name>_<mode>.json, with per-question
outcomes so the modes can be compared question by question.

The model is loaded once. The script stops after the validation gate unless
the no-RAG answers agree with lm-eval's (use --force to continue anyway).

Example (project root):
    python rag/run_rag_eval.py --model quantized_models/naive_4bit --name naive_quant_4bit --lm-eval-samples results/samples/naive_quant_4bit_mmlu.json
"""

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_model import load_model_from
from rag.embedder import Embedder
from rag.hyde import generate_hypothetical_passages
from rag.mcq import LETTERS, load_questions
from rag.retriever import Retriever
from rag.scorer import build_prompt, score_prompts


def accuracy_summary(questions, preds):
    flags = [int(p == q["answer"]) for q, p in zip(questions, preds)]
    n = len(flags)
    acc = sum(flags) / n
    per_subject = {}
    for q, f in zip(questions, flags):
        per_subject.setdefault(q["subject"], []).append(f)
    return {"accuracy": acc, "stderr": math.sqrt(acc * (1 - acc) / (n - 1)), "n": n,
            "per_subject": {s: sum(v) / len(v) for s, v in per_subject.items()}}, flags


def paired_summary(before, after):
    """Question-by-question comparison: how many answers got fixed, how many broke, and an exact sign test."""
    gained = sum(1 for b, a in zip(before, after) if a == 1 and b == 0)
    lost = sum(1 for b, a in zip(before, after) if a == 0 and b == 1)
    n = gained + lost
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(gained, lost) + 1)) / 2 ** n)
    return {"fixed": gained, "broken": lost, "net": gained - lost, "sign_test_p": p}


def retrieve(retriever, embedder, texts, is_query, k):
    return retriever.search(embedder.encode(texts, is_query=is_query), k=k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--lm-eval-samples", default=None, help="results/samples/<model>_mmlu.json from Phase 2")
    parser.add_argument("--k", type=int, default=3, help="passages added to the prompt")
    parser.add_argument("--modes", nargs="+", default=["none", "plain", "hyde"],
                        help="none, plain, hyde, and hyde_gen (only write the made-up passages)")
    parser.add_argument("--gen-batch", type=int, default=32)
    parser.add_argument("--score-batch", type=int, default=16)
    parser.add_argument("--hyde-tokens", type=int, default=80)
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke tests)")
    parser.add_argument("--force", action="store_true", help="continue even if validation fails")
    parser.add_argument("--redo", action="store_true", help="recompute the no-RAG scores instead of reusing the saved ones")
    args = parser.parse_args()

    os.makedirs("results/rag", exist_ok=True)
    questions = load_questions()
    if args.limit:
        questions = questions[:args.limit]
    keys = [f"{q['subject']}/{q['doc_id']}" for q in questions]

    print(f"Loading {args.model} ...", flush=True)
    model, tokenizer = load_model_from(args.model)
    out_path = lambda mode: f"results/rag/{args.name}_{mode}.json"
    flags_by_mode = {}

    def finish(mode, preds, extra):
        summary, flags = accuracy_summary(questions, preds)
        flags_by_mode[mode] = flags
        record = {"model": args.model, "mode": mode, "k": args.k, **summary,
                  "predictions": preds, "flags": flags, **extra}
        if mode != "none" and "none" in flags_by_mode:
            record["vs_no_rag"] = paired_summary(flags_by_mode["none"], flags)
        json.dump(record, open(out_path(mode), "w"))
        msg = f"[{mode}] accuracy {summary['accuracy']:.3f} +- {summary['stderr']:.3f}"
        if "vs_no_rag" in record:
            v = record["vs_no_rag"]
            msg += f" | vs no-RAG: fixed {v['fixed']}, broken {v['broken']}, sign-test p={v['sign_test_p']:.3f}"
        print(msg, flush=True)

    # ---- 1. no retrieval + validation gate -------------------------------------------
    print("\n== mode: none ==", flush=True)
    if os.path.exists(out_path("none")) and not args.limit and not args.redo:
        cached = json.load(open(out_path("none")))
        flags_by_mode["none"] = cached["flags"]
        print(f"[none] reusing {out_path('none')}: accuracy {cached['accuracy']:.3f}", flush=True)
    else:
        scores = score_prompts(model, tokenizer, [build_prompt(tokenizer, q) for q in questions], args.score_batch)
        preds_none = [max(range(4), key=lambda i: s[i]) for s in scores]
        finish("none", preds_none, {})

    if args.lm_eval_samples and not args.limit:
        reference = json.load(open(args.lm_eval_samples))
        ref_flags = {f"{t.replace('mmlu_', '')}/{d}": acc for t, pairs in reference.items() for d, acc in pairs}
        same = sum(1 for k, f in zip(keys, flags_by_mode["none"]) if ref_flags[k] == f)
        agreement = same / len(keys)
        print(f"VALIDATION: our no-RAG correctness matches lm-eval on {same}/{len(keys)} questions ({agreement:.1%})", flush=True)
        if agreement < 0.97 and not args.force:
            print("Validation failed (< 97% agreement). Stopping; investigate the scorer before trusting RAG numbers.")
            return

    hyde_cache = f"results/rag/{args.name}_hyde_passages.json"
    if "hyde_gen" in args.modes:
        # The made-up passages need only the language model, not the library, so they can be
        # generated while the library is still being built. Results are cached for later.
        print("\n== generating HyDE passages only ==", flush=True)
        generate_hypothetical_passages(model, tokenizer, questions, hyde_cache,
                                       batch_size=args.gen_batch, max_new_tokens=args.hyde_tokens)
    if not {"plain", "hyde"} & set(args.modes):
        return

    embedder, retriever = Embedder(), Retriever()
    query_texts = [q["question"] + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, q["choices"])) for q in questions]

    def answer_with(mode, hits):
        prompts = [build_prompt(tokenizer, q, [p["text"] for p, _ in h]) for q, h in zip(questions, hits)]
        scores = score_prompts(model, tokenizer, prompts, args.score_batch)
        preds = [max(range(4), key=lambda i: s[i]) for s in scores]
        finish(mode, preds, {"mean_top1_similarity": sum(h[0][1] for h in hits) / len(hits),
                             "retrieved_titles": [[p["title"] for p, _ in h] for h in hits]})

    # ---- 2. plain retrieval ------------------------------------------------------------
    if "plain" in args.modes:
        print("\n== mode: plain ==", flush=True)
        answer_with("plain", retrieve(retriever, embedder, query_texts, True, args.k))

    # ---- 3. HyDE -------------------------------------------------------------------------
    if "hyde" in args.modes:
        print("\n== mode: hyde ==", flush=True)
        passages = generate_hypothetical_passages(model, tokenizer, questions, hyde_cache,
                                                  batch_size=args.gen_batch, max_new_tokens=args.hyde_tokens)
        hits = retrieve(retriever, embedder, [passages[k] for k in keys], False, args.k)  # passages are embedded WITHOUT the query prefix
        answer_with("hyde", hits)


if __name__ == "__main__":
    main()
