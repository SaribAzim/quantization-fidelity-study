"""
Records every model's real answers for the website's arena.

Each model answers each prompt in deployment/prompts.json with greedy decoding
(always the most likely next token, so the answer is deterministic). We record
every token as it is produced, with its time, so the website can replay the
answer at the model's real speed. Then each answer is compared word by word
with the original model's answer, so the site can highlight where a quantized
model departs from the original.

The packed 4 bit models run fully on the GPU. The original 16 bit model does
not fit and runs offloaded (about 0.3 tokens a second), so it is recorded once
here and only replayed on the site.

Per model results are cached in results/gallery/<id>.json, so the run can be
stopped and resumed. Assemble only (no GPU):  python deployment/build_gallery.py --assemble

Run from the project root:   python deployment/build_gallery.py [--models id id ...] [--max-new-tokens 80]
"""

import argparse
import difflib
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deployment.catalog import CATALOG, ORIGINAL_REPO, by_id
from deployment.generation import stream_answer

PROMPTS = "deployment/prompts.json"
CACHE_DIR = "results/gallery"
OUT = "web/public/data/gallery.json"


def load(model_id):
    m = by_id(model_id)
    t0 = time.time()
    if model_id == "original":
        from load_model import load_model_from
        model, tokenizer = load_model_from(ORIGINAL_REPO)
    else:
        from quantization.packed import load_lora_state, load_packed_model
        model, tokenizer = load_packed_model(m["folder"])
        if m.get("adapters"):
            load_lora_state(model, torch.load(m["adapters"]))
    return model, tokenizer, time.time() - t0


def answer(model, tokenizer, prompt, max_new_tokens):
    """Greedy decoding, one token at a time (deployment/generation.py), recording when each token arrives."""
    out_ids, pieces, times = [], [], []
    for token, new_text, seconds in stream_answer(model, tokenizer, prompt, max_new_tokens):
        out_ids.append(token)
        pieces.append(new_text)
        times.append(seconds)
    decode_s = times[-1] - times[0] if len(times) > 1 else 0.0
    return {"text": "".join(pieces), "tokenIds": out_ids, "pieces": pieces, "times": times,
            "firstTokenS": times[0] if times else None,
            "tokensPerSecond": round((len(times) - 1) / decode_s, 3) if decode_s > 0 else None}


def record(model_id, prompts, max_new_tokens):
    path = f"{CACHE_DIR}/{model_id}.json"
    cache = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {"answers": {}}
    todo = [p for p in prompts if p["id"] not in cache["answers"]]
    if not todo:
        print(f"{model_id}: all {len(prompts)} answers cached", flush=True)
        return
    model, tokenizer, load_s = load(model_id)
    cache.setdefault("loadSeconds", round(load_s, 1))
    print(f"{model_id}: loaded in {load_s:.0f}s, {len(todo)} prompts to answer", flush=True)
    for p in todo:
        a = answer(model, tokenizer, p["prompt"], max_new_tokens)
        cache["answers"][p["id"]] = a
        json.dump(cache, open(path, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"  {p['id']}: {len(a['pieces'])} tokens, {a['tokensPerSecond']} tokens/s", flush=True)
    del model
    torch.cuda.empty_cache()


def word_diff(original, other):
    """Splits `other` into runs that match the original's wording or differ from it: [[text, same], ...]."""
    tokenize = lambda s: re.findall(r"\s+|\w+|[^\w\s]", s)
    a, b = tokenize(original), tokenize(other)
    runs = []
    for tag, _, _, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if j2 > j1:
            same = tag == "equal"
            text = "".join(b[j1:j2])
            if runs and runs[-1][1] == same:
                runs[-1][0] += text
            else:
                runs.append([text, same])
    return runs


def assemble(prompts):
    caches = {m["id"]: json.load(open(f"{CACHE_DIR}/{m['id']}.json", encoding="utf-8"))
              for m in CATALOG if m["gallery"] and os.path.exists(f"{CACHE_DIR}/{m['id']}.json")}
    assert "original" in caches, "record the original model first"
    gallery = {"prompts": [], "models": {k: {"loadSeconds": v.get("loadSeconds")} for k, v in caches.items()}}
    for p in prompts:
        ref = caches["original"]["answers"].get(p["id"])
        if not ref:
            continue
        entry = {**p, "answers": {}}
        for model_id, cache in caches.items():
            a = cache["answers"].get(p["id"])
            if not a:
                continue
            ref_ids, ids = ref["tokenIds"], a["tokenIds"]
            first_diff = next((i for i, (x, y) in enumerate(zip(ref_ids, ids)) if x != y), min(len(ref_ids), len(ids)))
            similarity = difflib.SequenceMatcher(a=ref["text"].split(), b=a["text"].split(), autojunk=False).ratio()
            entry["answers"][model_id] = {
                "text": a["text"], "pieces": a["pieces"], "times": a["times"], "tokensPerSecond": a["tokensPerSecond"],
                "identicalTokens": first_diff, "wordSimilarity": round(similarity, 3),
                "diff": [[a["text"], True]] if model_id == "original" else word_diff(ref["text"], a["text"]),
            }
        gallery["prompts"].append(entry)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(gallery, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {OUT}: {len(gallery['prompts'])} prompts x {len(caches)} models, {os.path.getsize(OUT) / 1024:.0f} KB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None, help="ids from deployment/catalog.py (default: all gallery models, original last)")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--limit", type=int, default=None, help="only the first N prompts (smoke test)")
    parser.add_argument("--assemble", action="store_true", help="only build web/public/data/gallery.json from the cache")
    args = parser.parse_args()

    prompts = json.load(open(PROMPTS, encoding="utf-8"))[:args.limit]
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not args.assemble:
        ids = args.models or [m["id"] for m in CATALOG if m["gallery"] and m["id"] != "original"] + ["original"]
        for model_id in ids:
            record(model_id, prompts, args.max_new_tokens)
    if os.path.exists(f"{CACHE_DIR}/original.json"):
        assemble(prompts)


if __name__ == "__main__":
    main()
