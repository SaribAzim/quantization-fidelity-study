"""
Exports everything the website shows into one file, web/public/data/results.json,
straight from results/*.json, so the site never contains a hand-typed number.

Run from the project root:   python deployment/export_site_data.py
"""

import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deployment.catalog import BASELINE_ID, CATALOG
from evaluation.plot_results import load, load_window_results

OUT = "web/public/data/results.json"
SUBJECT_LABELS = {"mmlu_high_school_computer_science": "High school computer science",
                  "mmlu_college_computer_science": "College computer science",
                  "mmlu_computer_security": "Computer security", "mmlu_machine_learning": "Machine learning"}
TYPE_LABELS = {"self_attn.q_proj": "q", "self_attn.k_proj": "k", "self_attn.v_proj": "v", "self_attn.o_proj": "o",
               "mlp.gate_proj": "gate", "mlp.up_proj": "up", "mlp.down_proj": "down"}
DASHES = ("-", "–", "—")


def metrics(r, is_original):
    ext = r["mmlu_extended_cs"]
    estimated = "packed_size_gb_estimate" in r
    return {
        "perplexity": round(r["perplexity"], 3),
        "kl": None if is_original else round(r["kl_divergence"], 4),
        "top1": 1.0 if is_original else round(r["top1_agreement"], 4),
        "exam50": r["mmlu_accuracy"],
        "exam": round(ext["accuracy"], 4),
        "examStderr": round(ext["stderr"], 4),
        "perSubject": {SUBJECT_LABELS[t]: round(v["accuracy"], 4) for t, v in ext["per_task"].items()},
        "sizeGb": round(r.get("packed_size_gb_estimate", r["model_size_gb"]), 2),
        "sizeMeasured": not estimated,
        # Speed and GPU memory are only meaningful for models that run fully on the GPU.
        "peakVramGb": round(r["peak_vram_gb"], 2) if r.get("fully_on_gpu") else None,
        "tokensPerSecond": round(r["tokens_per_second"], 2) if r.get("fully_on_gpu") else None,
    }


def main():
    models = []
    for m in CATALOG:
        r = load(m["result"])
        assert r, f"missing results/{m['result']}.json"
        entry = {k: m[k] for k in ("id", "label", "family", "runnable", "gallery", "featured", "note")}
        entry["metrics"] = metrics(r, m["id"] == "original")
        models.append(entry)

    by = {m["id"]: m["metrics"] for m in models}
    base, orig = by[BASELINE_ID], by["original"]
    for m in models:
        if m["id"] in ("original", BASELINE_ID):
            m["recovery"] = None
            continue
        x = m["metrics"]
        m["recovery"] = {"kl": round((base["kl"] - x["kl"]) / base["kl"], 4) if x["kl"] is not None else None,
                         "exam": round((x["exam"] - base["exam"]) / (orig["exam"] - base["exam"]), 4)}

    sensitivity = {}
    for key, suffix in (("perRow", ""), ("groups128", "_g128")):
        path = f"results/mixed/sensitivity{suffix}.json"
        if os.path.exists(path):
            s = json.load(open(path))
            sensitivity[key] = {"perLayer": [round(s["per_layer"][str(l)]["kl_change"], 5) for l in range(32)],
                                "perKind": {TYPE_LABELS[t]: round(v["kl_change"], 5) for t, v in s["per_type"].items()}}

    rag = {}
    for mode in ("none", "plain", "hyde"):
        r = load(f"rag/naive_quant_4bit_{mode}")
        rag[mode] = {"exam": round(r["accuracy"], 4), "stderr": round(r["stderr"], 4)}
        if "vs_no_rag" in r:
            rag[mode].update(fixed=r["vs_no_rag"]["fixed"], broken=r["vs_no_rag"]["broken"], p=round(r["vs_no_rag"]["sign_test_p"], 3))

    log = json.load(open("results/qlora/train_log.json"))["log"]
    qlora_curve = {"train": [[e["step"], round(e["train_loss"], 4)] for e in log if "train_loss" in e],
                   "validation": [[e["step"], round(e["val_loss"], 4)] for e in log if "val_loss" in e]}

    windows = load_window_results()
    window_labels = {"base": "Original", "quantized_models/naive_4bit_g128": "Groups of 128",
                     "quantized_models/gptq_4bit_g128": "GPTQ, 2048 token calibration",
                     "quantized_models/gptq_4bit_g128_cal512": "GPTQ, 512 token calibration"}
    gptq_window = [{"label": window_labels[k], "short": v[512], "long": v[2048]} for k, v in windows.items() if k in window_labels]

    site = {
        "generated": date.today().isoformat(),
        "facts": {"model": "Llama 3.1 8B Instruct", "gpu": "RTX 3060 Ti, 8 GB", "examQuestions": 412, "examNoise": 0.025,
                  "subjects": list(SUBJECT_LABELS.values()), "libraryPassages": 10751},
        "baselineId": BASELINE_ID,
        "models": models,
        "sensitivity": sensitivity,
        "rag": rag,
        "qlora": qlora_curve,
        "gptqWindow": gptq_window,
    }

    # The site's copy rule: no hyphens or dashes in any label or note shown on the page.
    for m in models:
        for field in ("label", "family", "featured", "note"):
            assert not any(d in (m[field] or "") for d in DASHES), f"dash in {m['id']}.{field}: {m[field]}"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(site, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"Wrote {OUT}: {len(models)} models, {os.path.getsize(OUT) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
