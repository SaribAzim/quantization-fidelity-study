"""
Turns every results/*.json into the project's figures and a summary table.
Nothing is typed by hand: re-running this after a new experiment updates all
of it. Missing results files are skipped, so it works at any stage.

Run from the project root:   python evaluation/plot_results.py
Writes: results/figures/*.png  and  results/summary.md
"""

import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = "results/figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
# One colour per FAMILY of method, fixed order (so a family keeps its colour in every figure).
FAMILY_COLOR = {"Original": "#2a78d6", "Naive, one ruler per row": "#1baf7a", "Naive, groups of 128": "#eda100",
                "GPTQ": "#e87ba4", "Mixed precision": "#6250d6", "QLoRA": "#e34948", "8-bit": "#eb6834"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})

# (label, results file, family)
MODELS = [
    ("Original (16-bit)", "baseline", "Original"),
    ("Naive 8-bit", "naive_quant_8bit", "8-bit"),
    ("Naive 4-bit, per row", "packed_naive_4bit", "Naive, one ruler per row"),
    ("Per row + QLoRA", "qlora_naive_4bit", "QLoRA"),
    ("Per row + 8-bit L0,1,31", "packed_mixed_top3", "Mixed precision"),
    ("Per row + 8-bit 8 layers", "packed_mixed_layers", "Mixed precision"),
    ("Per row + 8-bit v,k,o,down", "packed_mixed_types", "Mixed precision"),
    ("Naive 4-bit, g128", "packed_naive_4bit_g128", "Naive, groups of 128"),
    ("g128 + 8-bit L0,1,31", "packed_g128_mixed_top3", "Mixed precision"),
    ("g128 + 8-bit v,k,o,q", "packed_g128_mixed_types", "Mixed precision"),
    ("GPTQ g128 (cal 512)", "gptq_4bit_g128_cal512", "GPTQ"),
    ("GPTQ g128 (cal 2048)", "gptq_4bit_g128", "GPTQ"),
]


def load(name):
    path = f"results/{name}.json"
    return json.load(open(path)) if os.path.exists(path) else None


def size_gb(r):
    """Real size for packed models, packed-size estimate for fake-quantized ones, 16-bit size for the original."""
    return r.get("packed_size_gb_estimate", r["model_size_gb"])


def rows():
    out = []
    for label, name, family in MODELS:
        r = load(name)
        if r:
            out.append((label.strip(), label, family, r))
    return out


def finish(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(f"{FIG_DIR}/{name}.png", dpi=170)
    plt.close(fig)


def legend_for(fig, families):
    """One legend under the whole figure (so it never covers a bar or a point)."""
    handles = [plt.Line2D([0], [0], marker="s", ls="", color=FAMILY_COLOR[f], markersize=8, label=f) for f in families]
    fig.legend(handles=handles, frameon=False, fontsize=8, loc="outside lower center", ncol=len(families))


def overview_bars(data):
    """Exam accuracy and distance-from-original for every model (horizontal bars, grouped by family colour)."""
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 5.0), constrained_layout=True, sharey=True)
    labels = [d[1] for d in data][::-1]
    fams = [d[2] for d in data][::-1]
    ypos = list(range(len(labels)))
    acc = [d[3]["mmlu_extended_cs"]["accuracy"] * 100 for d in data][::-1]
    err = [d[3]["mmlu_extended_cs"]["stderr"] * 100 for d in data][::-1]
    kl = [d[3].get("kl_divergence") or 0.0 for d in data][::-1]
    cols = [FAMILY_COLOR[f] for f in fams]
    ax[0].barh(ypos, acc, xerr=err, color=cols, height=0.62, error_kw={"ecolor": INK2, "lw": 0.8, "capsize": 2})
    for y, v in enumerate(acc):
        ax[0].text(v + 3.2, y, f"{v:.1f}", va="center", fontsize=8, color=INK)
    original = [d for d in data if d[2] == "Original"]
    if original:
        ax[0].axvline(original[0][3]["mmlu_extended_cs"]["accuracy"] * 100, color=FAMILY_COLOR["Original"], lw=0.9, ls="--")
    ax[0].set_xlim(40, 64); ax[0].set_title("CS exam accuracy, 412 questions (%)", fontsize=9, loc="left", color=INK)
    ax[1].barh(ypos, kl, color=cols, height=0.62)
    for y, v in enumerate(kl):
        ax[1].text(v + 0.012, y, "-" if v == 0 else f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax[1].set_xlim(0, 0.6); ax[1].set_title("Distance from the original's predictions (KL, lower = better)", fontsize=9, loc="left", color=INK)
    for a in ax:
        a.grid(axis="x", color=GRID, lw=0.6); a.set_axisbelow(True)
    ax[0].set_yticks(ypos); ax[0].set_yticklabels(labels)
    legend_for(fig, [f for f in FAMILY_COLOR if f in set(fams)])
    finish(fig, "overview")


POINT_LABELS = {  # label text and (dx, dy) offset in points, chosen to avoid overlaps
    "Original (16-bit)": ("Original", (-28, 8)), "Naive 8-bit": ("Naive 8-bit", (6, 6)),
    "Naive 4-bit, per row": ("Naive 4-bit per row", (8, -3)), "Per row + QLoRA": ("+ QLoRA", (6, 8)),
    "Per row + 8-bit L0,1,31": ("per row + L0,1,31", (8, -3)), "Per row + 8-bit 8 layers": ("per row + 8 layers", (8, -3)),
    "Per row + 8-bit v,k,o,down": ("per row + v,k,o,down", (8, 7)), "Naive 4-bit, g128": ("Naive g128", (-62, -17)),
    "g128 + 8-bit L0,1,31": ("g128 + L0,1,31", (8, -14)), "g128 + 8-bit v,k,o,q": ("g128 + v,k,o,q", (8, -3)),
    "GPTQ g128 (cal 512)": ("GPTQ g128", (-14, 9)), "GPTQ g128 (cal 2048)": ("GPTQ cal 2048", (8, -3)),
}


def size_vs_quality(data):
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9), constrained_layout=True)
    for label, raw_label, family, r in data:
        x = size_gb(r)
        ax[0].scatter(x, r["mmlu_extended_cs"]["accuracy"] * 100, s=60, color=FAMILY_COLOR[family], zorder=3)
        text, offset = POINT_LABELS.get(label, (label, (6, 4)))
        ax[0].annotate(text, (x, r["mmlu_extended_cs"]["accuracy"] * 100), textcoords="offset points", xytext=offset, fontsize=7.5, color=INK)
        if r.get("kl_divergence") is not None:
            ax[1].scatter(x, r["kl_divergence"], s=60, color=FAMILY_COLOR[family], zorder=3)
    ax[0].set_xlabel("size when packed (GB)"); ax[0].set_ylabel("CS exam accuracy (%)")
    ax[0].set_title("Higher is better", fontsize=9, loc="left", color=INK)
    ax[1].set_xlabel("size when packed (GB)"); ax[1].set_ylabel("KL from original")
    ax[1].set_yscale("symlog", linthresh=0.01); ax[1].set_ylim(0, 0.7)
    ax[1].set_title("Lower is better (log scale)", fontsize=9, loc="left", color=INK)
    for a in ax:
        a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
    legend_for(fig, [f for f in FAMILY_COLOR if f in {d[2] for d in data}])
    finish(fig, "size_vs_quality")


def sensitivity():
    for suffix, title in (("", "per-row 4-bit model"), ("_g128", "groups-of-128 4-bit model")):
        path = f"results/mixed/sensitivity{suffix}.json"
        if not os.path.exists(path):
            continue
        s = json.load(open(path))
        fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.0), constrained_layout=True, gridspec_kw={"width_ratios": [2.2, 1]})
        lk = [s["per_layer"][str(l)]["kl_change"] for l in range(32)]
        top = sorted(lk)[-3]
        ax[0].bar(range(32), lk, color=["#e34948" if v >= top else "#2a78d6" for v in lk], width=0.75)
        for l, v in enumerate(lk):
            if v >= top:
                ax[0].text(l, v * 1.02, f"layer {l}", ha="center", fontsize=7.5, color=INK)
        ax[0].set_xlabel("decoder layer"); ax[0].set_ylabel("output change when upgraded to 8-bit")
        ax[0].set_title(f"Which layers suffer most from 4-bit rounding ({title})", fontsize=9, loc="left", color=INK)
        types = list(s["per_type"])
        ax[1].bar(["q", "k", "v", "o", "gate", "up", "down"], [s["per_type"][t]["kl_change"] for t in types], color="#6250d6", width=0.65)
        ax[1].set_title("By kind of table", fontsize=9, loc="left", color=INK)
        for a in ax:
            a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True)
        finish(fig, f"sensitivity{suffix}")


def rag():
    none, plain, hyde = (load(f"rag/naive_quant_4bit_{m}") for m in ("none", "plain", "hyde"))
    base = load("baseline")
    if not (none and plain and hyde and base):
        return
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.0), constrained_layout=True)
    names = ["Original\nno RAG", "Naive 4-bit\nno RAG", "Naive 4-bit\nplain RAG", "Naive 4-bit\nHyDE RAG"]
    vals = [base["mmlu_extended_cs"]["accuracy"], none["accuracy"], plain["accuracy"], hyde["accuracy"]]
    errs = [base["mmlu_extended_cs"]["stderr"], none["stderr"], plain["stderr"], hyde["stderr"]]
    ax[0].bar(names, [v * 100 for v in vals], yerr=[e * 100 for e in errs], capsize=3, width=0.58,
              color=["#2a78d6", "#1baf7a", "#eb6834", "#eb6834"], error_kw={"ecolor": INK2, "lw": 0.9})
    for i, v in enumerate(vals):
        ax[0].text(i, v * 100 + errs[i] * 100 + 1, f"{v * 100:.1f}", ha="center", fontsize=8.5, color=INK)
    ax[0].set_ylim(0, 70); ax[0].set_title("Retrieval did not recover the loss", fontsize=9, loc="left", color=INK)
    for i, (name, m) in enumerate([("Plain RAG", plain), ("HyDE RAG", hyde)]):
        v = m["vs_no_rag"]
        ax[1].barh(i, v["fixed"], color="#1baf7a", height=0.45); ax[1].barh(i, -v["broken"], color="#e34948", height=0.45)
        ax[1].text(v["fixed"] + 1, i, f"fixed {v['fixed']}", va="center", fontsize=8.5, color=INK)
        ax[1].text(-v["broken"] - 1, i, f"broken {v['broken']}", va="center", ha="right", fontsize=8.5, color=INK)
    ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(["Plain RAG", "HyDE RAG"]); ax[1].axvline(0, color=INK2, lw=0.8)
    ax[1].set_xlim(-95, 95); ax[1].set_title("Answers changed vs no retrieval", fontsize=9, loc="left", color=INK)
    for a in ax:
        a.grid(axis="y" if a is ax[0] else "x", color=GRID, lw=0.6); a.set_axisbelow(True)
    finish(fig, "rag")


def qlora():
    path = "results/qlora/train_log.json"
    base, packed, q = load("baseline"), load("packed_naive_4bit"), load("qlora_naive_4bit")
    if not (os.path.exists(path) and base and packed and q):
        return
    log = json.load(open(path))["log"]
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.0), constrained_layout=True)
    ax[0].plot([e["step"] for e in log if "train_loss" in e], [e["train_loss"] for e in log if "train_loss" in e], color="#c9c8c2", lw=1.1, label="training loss")
    ax[0].plot([e["step"] for e in log if "val_loss" in e], [e["val_loss"] for e in log if "val_loss" in e], color="#6250d6", lw=2, marker="o", ms=3.5, label="validation loss")
    ax[0].set_xlabel("training step"); ax[0].set_ylabel("loss"); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("QLoRA learns the text well...", fontsize=9, loc="left", color=INK)
    names, cols = ["Original", "RTN 4-bit", "+ QLoRA"], ["#2a78d6", "#1baf7a", "#e34948"]
    kl = [0, packed["kl_divergence"], q["kl_divergence"]]
    ax[1].bar(names, kl, color=cols, width=0.58)
    for i, v in enumerate(kl):
        ax[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)
    ax[1].set_title("...but drifts away from the original (KL, lower = better)", fontsize=9, loc="left", color=INK)
    for a in ax:
        a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
    finish(fig, "qlora")


def gptq_window():
    out = {}
    for fn in os.listdir("logs"):
        if fn.startswith("chain_eval_"):
            raw = open(os.path.join("logs", fn), "rb").read()
            text = raw.decode("utf-16", errors="ignore") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig", errors="ignore")
            for line in text.splitlines():
                m = re.match(r"RESULT (\S+) window (\d+): perplexity ([\d.]+)", line)
                if m:
                    out.setdefault(m.group(1), {})[int(m.group(2))] = float(m.group(3))
    keys = [("Original", "base"), ("Naive g128", "quantized_models/naive_4bit_g128"), ("GPTQ, calibrated\non 2048-token text", "quantized_models/gptq_4bit_g128"),
            ("GPTQ, calibrated\non 512-token text", "quantized_models/gptq_4bit_g128_cal512")]
    keys = [(n, k) for n, k in keys if k in out and 512 in out[k] and 2048 in out[k]]
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(9.2, 2.8), constrained_layout=True)
    x = range(len(keys))
    b1 = ax.bar([i - 0.19 for i in x], [out[k][512] for _, k in keys], 0.36, color="#2a78d6", label="short test windows (512 tokens)")
    b2 = ax.bar([i + 0.19 for i in x], [out[k][2048] for _, k in keys], 0.36, color="#eb6834", label="long test windows (2048 tokens)")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{bar.get_height():.1f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(list(x)); ax.set_xticklabels([n for n, _ in keys], fontsize=8); ax.set_ylim(0, 23); ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("GPTQ overfits to the length of its calibration text (perplexity, lower = better)", fontsize=9, loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    finish(fig, "gptq_window")


SCORE_ROWS = ["Original (16-bit)", "Naive 8-bit", "Naive 4-bit, per row", "Per row + QLoRA", "Per row + 8-bit v,k,o,down",
              "Naive 4-bit, g128", "g128 + 8-bit L0,1,31", "GPTQ g128 (cal 512)"]


def scoreboard(data):
    """A compact table as a picture: each metric column is shaded from light (worse) to blue (better); the best value is bold."""
    from matplotlib.colors import to_rgb
    rows_ = [d for d in data if d[1] in SCORE_ROWS]
    columns = [  # title, value function, format, which direction is better (None = not shaded)
        ("Size (GB)", lambda r: size_gb(r), "{:.2f}", None),
        ("Perplexity", lambda r: r["perplexity"], "{:.2f}", "low"),
        ("KL from original", lambda r: r.get("kl_divergence") or 0.0, "{:.3f}", "low"),
        ("Same top word", lambda r: 1.0 if r.get("top1_agreement") is None else r["top1_agreement"], "{:.1%}", "high"),
        ("CS exam", lambda r: r["mmlu_extended_cs"]["accuracy"], "{:.1%}", "high"),
    ]
    n, light, deep = len(rows_), to_rgb("#f2f5f9"), to_rgb("#8ab6ec")
    fig, ax = plt.subplots(figsize=(9.6, 0.43 * (n + 2.1)))
    ax.set_xlim(0, 9.6); ax.set_ylim(n + 1.25, -0.2); ax.axis("off")
    x0, widths = 2.75, [1.1, 1.3, 1.65, 1.5, 1.2]
    xs = [x0 + sum(widths[:i]) for i in range(len(widths))]
    for (title, *_), x, w in zip(columns, xs, widths):
        ax.text(x + w / 2, 0.15, title, ha="center", va="center", fontsize=8.6, color=INK2, fontweight="bold")
    ax.plot([0, 9.6], [0.55, 0.55], color="#cfcec8", lw=0.9)
    for ci, (title, fn, fmt, better) in enumerate(columns):
        vals = [fn(d[3]) for d in rows_]
        lo, hi = min(vals), max(vals)
        best = (min(vals) if better == "low" else max(vals)) if better else None
        for ri, (d, v) in enumerate(zip(rows_, vals)):
            y = ri + 1.1
            if better:
                s = (v - lo) / (hi - lo) if hi > lo else 0.5
                s = 1 - s if better == "low" else s
                face = tuple(light[k] + (deep[k] - light[k]) * s for k in range(3))
                ax.add_patch(plt.Rectangle((xs[ci] + 0.04, y - 0.4), widths[ci] - 0.08, 0.8, color=face, lw=0))
            text = fmt.format(v)
            if title == "KL from original" and d[3].get("kl_divergence") is None:
                text = "-"
            if title == "Size (GB)" and "packed_size_gb_estimate" in d[3]:
                text += "*"
            if title == "Perplexity" and d[1] == "Per row + QLoRA":
                text += "†"  # below the original's: the adapters learned the text's style (see footnote)
            ax.text(xs[ci] + widths[ci] / 2, y, text, ha="center", va="center", fontsize=8.8, color=INK,
                    fontweight="bold" if (better and v == best) else "normal")
    for ri, d in enumerate(rows_):
        ax.text(0.05, ri + 1.1, d[1], ha="left", va="center", fontsize=8.8, color=INK)
        if ri < n - 1:
            ax.plot([0, 9.6], [ri + 1.5, ri + 1.5], color="#ececE8", lw=0.6)
    ax.text(0.05, n + 0.62, "* size when packed (estimate); other sizes are measured.   Shading: darker blue = better within each column; bold = best.",
            fontsize=7.4, color=INK2, va="center")
    ax.text(0.05, n + 0.95, "† lower than the original's perplexity: QLoRA learned the training text's style, and moved further from the original (KL).",
            fontsize=7.4, color=INK2, va="center")
    finish(fig, "scoreboard")


def pipeline():
    """A minimal diagram of the study's pipeline."""
    from matplotlib.patches import FancyBboxPatch
    stages = [("Original model", "Llama 3.1 8B Instruct\n16-bit, 15 GB", "#2a78d6"),
              ("Quantize", "round to nearest\n(per row, g128) or GPTQ", "#1baf7a"),
              ("Pack to 4-bit", "5.2 to 6.1 GB, fully\non an 8 GB GPU", "#eda100"),
              ("Repair", "mixed precision, QLoRA,\nRAG and HyDE", "#6250d6"),
              ("Measure", "perplexity, KL, same top\nword, CS exam", "#eb6834")]
    fig, ax = plt.subplots(figsize=(9.6, 1.75))
    ax.set_xlim(0, 9.6); ax.set_ylim(0, 1.75); ax.axis("off")
    w, gap = 1.62, 0.3
    for i, (head, sub, color) in enumerate(stages):
        x = 0.3 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.2), w, 1.3, boxstyle="round,pad=0,rounding_size=0.06", fc="white", ec="#d3d2cb", lw=1.0))
        ax.add_patch(plt.Rectangle((x + 0.06, 1.4), w - 0.12, 0.05, color=color, lw=0))
        ax.text(x + w / 2, 1.15, head, ha="center", va="center", fontsize=9.4, fontweight="bold", color=INK)
        ax.text(x + w / 2, 0.68, sub, ha="center", va="center", fontsize=7.0, color=INK2, linespacing=1.5)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + w + gap - 0.03, 0.85), xytext=(x + w + 0.03, 0.85), arrowprops=dict(arrowstyle="->", color="#9a9992", lw=1.0))
    finish(fig, "pipeline")


def summary_table(data):
    lines = ["# Results summary", "", "Generated by `evaluation/plot_results.py` from `results/*.json`. Exam = 412 CS MMLU questions "
             "(uncertainty about plus or minus 2.5 points). Size is the real size for packed models and the packed-size estimate for fake-quantized ones.", "",
             "| Model | Perplexity | KL from original | Same top word | Exam (50 q) | Exam (412 q) | Size (GB) |", "|---|---|---|---|---|---|---|"]
    for label, raw_label, _, r in data:
        kl = r.get("kl_divergence"); t1 = r.get("top1_agreement")
        lines.append(f"| {raw_label} | {r['perplexity']:.2f} | "
                     f"{'-' if kl is None else f'{kl:.3f}'} | {'100%' if t1 is None else f'{t1 * 100:.1f}%'} | {r['mmlu_accuracy'] * 100:.0f}% | "
                     f"{r['mmlu_extended_cs']['accuracy'] * 100:.1f}% | {size_gb(r):.2f} |")
    open("results/summary.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    data = rows()
    overview_bars(data)
    size_vs_quality(data)
    sensitivity()
    rag()
    qlora()
    gptq_window()
    scoreboard(data)
    pipeline()
    summary_table(data)
    print(f"{len(data)} models plotted; figures in {FIG_DIR}/, table in results/summary.md")
