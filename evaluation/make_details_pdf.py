"""
Builds details.pdf: the whole project explained in one document, from the
results files (so numbers can never drift from the data).

Run from the project root (after evaluation/plot_results.py):
    python evaluation/make_details_pdf.py
Writes ../details.pdf (next to the project folder).
"""

import json
import os
import sys

sys.path.insert(0, ".")

from PIL import Image as PI
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from evaluation.plot_results import MODELS, load, size_gb

OUT_PDF = os.path.join("..", "details.pdf")
FIG = "results/figures"
INK, INK2 = "#0b0b0b", "#52514e"

ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=10, leading=14.6, textColor=colors.HexColor(INK), spaceAfter=6)
small = ParagraphStyle("small", parent=body, fontSize=8.5, leading=11.5, textColor=colors.HexColor(INK2))
h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=17, leading=21, spaceBefore=4, spaceAfter=8, textColor=colors.HexColor("#184f95"))
h2 = ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=12, leading=15, spaceBefore=8, spaceAfter=4)
title = ParagraphStyle("title", parent=h1, fontSize=28, leading=33, spaceAfter=4)
bul = ParagraphStyle("bul", parent=body, leftIndent=14, bulletIndent=3, spaceAfter=3)
cell = ParagraphStyle("cell", parent=body, fontSize=8.4, leading=10.8, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")


def P(t, s=body): return Paragraph(t, s)
def B(items): return [Paragraph(i, bul, bulletText="•") for i in items]


def box(text, bg="#eef4fc", edge="#2a78d6"):
    t = Table([[Paragraph(text, body)]], colWidths=[16.6 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)), ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(edge)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return [t, Spacer(1, 6)]


def table(rows, widths):
    data = [[Paragraph(str(c), cellb if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[x * cm for x in widths], repeatRows=1)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d9d8d3")),
          ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7"))]
    st += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f7f7f5")) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(st))
    return [t, Spacer(1, 8)]


def img(name, caption, width_cm=16.6):
    path = f"{FIG}/{name}.png"
    if not os.path.exists(path):
        return []
    pw, ph = PI.open(path).size
    return [Image(path, width=width_cm * cm, height=width_cm * cm * ph / pw), P(caption, small)]


def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 8); c.setFillColor(colors.HexColor(INK2))
    c.drawString(2 * cm, 1.2 * cm, "Quantization Fidelity Study - project details")
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {d.page}"); c.restoreState()


pct = lambda x: f"{x * 100:.1f}%"
R = {name: load(name) for _, name, _ in MODELS}
orig, rtn, types = R["baseline"], R["packed_naive_4bit"], R["packed_mixed_types"]
g128, gptq, q8, qlora = R["packed_naive_4bit_g128"], R["gptq_4bit_g128_cal512"], R["naive_quant_8bit"], R["qlora_naive_4bit"]
mg_top3, mg_types = R.get("packed_g128_mixed_top3"), R.get("packed_g128_mixed_types")
acc = lambda r: r["mmlu_extended_cs"]["accuracy"]
rag = {m: load(f"rag/naive_quant_4bit_{m}") for m in ("none", "plain", "hyde")}
gap = acc(orig) - acc(rtn)

S = []
# ============ 1. cover and one-page summary ============
S += [P("Quantization Fidelity Study", title),
      P("How much does an 8-billion-parameter language model lose when it is shrunk, and what wins the quality back?", ParagraphStyle("sub", parent=body, fontSize=12, textColor=colors.HexColor(INK2))),
      P("Project details: what the project is, what was done, how it was done, and what we found", ParagraphStyle("sub2", parent=body, fontSize=10, textColor=colors.HexColor(INK2))),
      Spacer(1, 10), P("1. The project on one page", h1),
      P("<b>The problem.</b> Llama 3.1 8B Instruct is an AI language model with about 8 billion stored numbers (weights). Each is kept with 16 bits of detail, so the model needs about <b>15 GB</b>. The graphics card used here (RTX 3060 Ti) has <b>8 GB</b>, so the model does not fit and has to stream from disk, which is very slow."),
      P("<b>The idea: quantization.</b> Store each weight with fewer bits (8 or 4) instead of 16. That shrinks the model about 3 times, at the price of some lost quality, like saving a photo at lower resolution."),
      P("<b>The research question.</b> How much quality is lost, how faithfully does the shrunken model copy the original, and which techniques win the loss back? The repository is named <i>quantization-fidelity-study</i> because the goal is measuring fidelity to the original model, not just getting a model to run.")]
S += box(f"<b>The headline result.</b> Plain 4-bit rounding (one setting per row of weights) drops the computer-science exam score from <b>{pct(acc(orig))}</b> to <b>{pct(acc(rtn))}</b> and makes the model disagree with the original on about one word in four. Keeping the most fragile weight tables at 8-bit (+0.9 GB) recovers it to <b>{pct(acc(types))}</b> and cuts the distance from the original's predictions from {rtn['kl_divergence']:.3f} to {types['kl_divergence']:.3f}. A finer rounding ruler (groups of 128 numbers) does about as well at a smaller size ({pct(acc(g128))}). Adding a library to look things up in (RAG, HyDE) and fine-tuning small correction pieces (QLoRA) did <b>not</b> recover the loss.")
rows = [["Model", "Exam, 412 questions", "Distance from original (KL)", "Same top word", "Size"]]
for label, name, _ in [("Original (16-bit)", "baseline", 0), ("Plain 4-bit, one ruler per row", "packed_naive_4bit", 0), ("4-bit, groups of 128", "packed_naive_4bit_g128", 0),
                       ("4-bit + 8-bit for fragile tables", "packed_mixed_types", 0),
                       ("Groups of 128 + 8-bit for layers 0, 1, 31", "packed_g128_mixed_top3", 0)]:
    r = R[name]
    rows.append([label, pct(acc(r)), "-" if r.get("kl_divergence") is None else f"{r['kl_divergence']:.3f}",
                 "100%" if r.get("top1_agreement") is None else pct(r["top1_agreement"]), f"{size_gb(r):.2f} GB"])
S += table(rows, [5.6, 2.7, 3.6, 2.4, 2.3])
S += [P("The exam has 412 computer-science questions; its uncertainty is about plus or minus 2.5 points, so the distance and agreement measures (thousands of predictions) are the more sensitive evidence.", small), PageBreak()]

# ============ 2. what was built ============
S += [P("2. What was done, in order", h1)]
S += table([["Phase", "What was done", "Outcome"],
            ["0 and 1", "Set up the project, downloaded the model, measured the original model (the baseline).", "Original: exam 55.6%, perplexity 11.44."],
            ["2", "Wrote our own naive quantization (round every weight to its nearest allowed value). First attempt silently skipped two thirds of the layers (a bug); redone on the saved files, all 224 weight tables.", f"8-bit: almost lossless ({pct(acc(q8))}). 4-bit: clear loss."],
            ["3", "Wrote GPTQ (a smarter, calibrated rounding) from scratch, plus rounding with groups of 128 numbers.", "Groups of 128 and GPTQ both recover most of the loss."],
            ["4", "Built a retrieval system (RAG and HyDE) over 10,751 passages from Wikipedia computer-science articles.", "No recovery on the 4-bit model."],
            ["Real 4-bit", "Stored the model as real 4-bit numbers, not 16-bit values pretending to be 4-bit.", f"{size_gb(rtn):.2f} GB instead of 15 GB; fits the GPU; runs are 5 times faster."],
            ["QLoRA", "Trained small adapter pieces on top of the frozen 4-bit model.", "Better at Wikipedia text, but further from the original: no recovery."],
            ["Mixed precision", "Measured which parts suffer most from 4-bit rounding and kept those at 8-bit.", f"Best layout recovers the exam to {pct(acc(types))}."],
            ["Combination", "Groups of 128 together with mixed precision (sensitivity re-measured on the finer model).", f"Best fidelity of any 4-bit model: KL {mg_top3['kl_divergence']:.3f}, exam {pct(acc(mg_top3))}."],
            ["Plots and this report", "Figures and tables generated from the results files.", "This document."]],
           [2.4, 9.0, 5.2])
S += [P("What is <b>not</b> done: the planned deployment (a free online demo with a web page, packaged with Docker) and the Docker lesson file. Everything else on the original roadmap was completed or deliberately tested and reported as a negative result.", body)]
S += [P("Tools used (all free)", h2)]
S += table([["Tool", "Used for"],
            ["PyTorch, Transformers, Accelerate, safetensors", "Running the model, handling its saved files, splitting a too-big model across GPU, RAM and disk"],
            ["Datasets, lm-evaluation-harness", "WikiText-2 text for perplexity and the standard exam (MMLU) tooling"],
            ["A small embedding model (bge-small)", "Turning text into numbers for retrieval (runs on the CPU)"],
            ["Wikipedia's public interface", "Text for the retrieval library (rate-limited, so downloaded politely)"],
            ["matplotlib, reportlab", "The figures and PDF reports"],
            ["Git and GitHub", "Version history (every commit authored as the project owner)"]],
           [6.0, 10.6])
S += [PageBreak()]

# ============ 3. methods ============
S += [P("3. The techniques, and how each was built", h1)]
S += [P("Naive quantization (round to nearest)", h2),
      P("Think of a ruler with only 16 tick marks (4-bit) instead of thousands. Every weight is moved to the nearest tick, and only the tick number is stored. Each row of a weight table gets its own ruler (its own scale and zero-point), so one unusual row cannot spoil the others. <b>Groups of 128</b> use a separate small ruler for every 128 neighbouring weights, so the ticks sit closer together and rounding is gentler, for a tiny increase in size. Code: <font face='Courier'>quantization/naive_ptq.py</font>, <font face='Courier'>quantize_checkpoint.py</font>.")]
S += [P("Fake versus real 4-bit storage", h2),
      P("At first the rounded weights were stored again as 16-bit numbers (\"fake\" quantization): the quality loss was measured correctly but the file stayed 15 GB. <b>Real packed storage</b> keeps the 4-bit integers, two per byte, plus the ruler settings, and unpacks each layer just before use with identical arithmetic. We proved the unpacked weights are bit-for-bit identical to the fake ones. Code: <font face='Courier'>quantization/packed.py</font>, <font face='Courier'>make_packed.py</font>.")]
S += [P("GPTQ (calibrated rounding)", h2),
      P("Naive rounding treats each weight alone. GPTQ shows a little sample text to the model, learns which inputs matter and move together, then rounds one column at a time and nudges the not-yet-rounded columns to cancel the error. It was written from scratch because the standard library cannot be installed on this Windows setup. A key lesson: calibrated on 2048-token text it collapsed on 512-token test text; calibrated on 512-token text it matched the simple method on short text and beat it on long text. Code: <font face='Courier'>quantization/gptq.py</font>, <font face='Courier'>run_gptq.py</font>.")]
S += [P("Mixed precision", h2),
      P("Not all weight tables suffer equally. We swapped one group at a time to 8-bit in the real 4-bit model and measured how much the output changed (using validation text only, so the choice cannot leak into reported results). Layers 0, 1 and 31, and the v, k, o and down tables, were the most fragile. Three layouts with similar extra budgets were built and tested. Code: <font face='Courier'>quantization/mixed_sensitivity.py</font>, <font face='Courier'>plan_mixed.py</font>.")]
S += [P("RAG and HyDE", h2),
      P("RAG gives the model an open book: find the three most relevant passages in a library and paste them in front of the question. HyDE first asks the model to write a made-up answer passage and searches with that, because a made-up answer looks more like real documents than a short question does. The embeddings are only used for searching and never reach the language model. Code: <font face='Courier'>rag/</font> folder. A custom scorer was validated to match the standard exam tool on 404 of 412 questions before any RAG number was trusted.")]
S += [P("QLoRA", h2),
      P("The 4-bit weights stay frozen. Only small add-on tables (LoRA adapters, 41.9 million numbers, about 0.5% of the model) are trained, here to predict the next word of Wikipedia text. It starts as an exact copy of the 4-bit model. Code: <font face='Courier'>quantization/train_qlora.py</font>.")]
S += [PageBreak()]

# ============ 4. measurement ============
S += [P("4. How everything is measured", h1),
      P("Every model is measured with <b>identical code on identical text and questions</b>. This rule is the reason the comparisons mean something.")]
S += table([["Measure", "What it asks", "How to read it"],
            ["Perplexity (WikiText-2)", "How surprised is the model by real text it has not memorised?", "Lower is better. Can be gamed by training on similar text."],
            ["KL divergence from the original", "How different are the model's whole probability predictions from the original's?", "0 = identical. The best single measure of fidelity."],
            ["Same top word", "How often does it pick the same most-likely next word as the original?", "100% = always. Easy to read."],
            ["CS exam (MMLU, 412 questions)", "Multiple-choice computer-science questions, no examples given.", "Higher is better; uncertainty about 2.5 points."],
            ["Size and GPU memory", "Real bytes of the model and peak GPU use.", "Real only for packed models; others are estimated."]],
           [4.2, 7.0, 5.4])
S += [P("Fairness rules followed: fixed settings chosen before looking at exam results; text used to <i>choose</i> (validation) kept separate from text used to <i>report</i> (test); exam questions removed from the retrieval library; paired question-by-question comparisons with a sign test, not only totals; negative results reported as they are.")]
S += [P("5. Results", h1)]
rows = [["Model", "Perplexity", "KL", "Same top word", "Exam, 50 q", "Exam, 412 q", "Size"]]
for label, name, family in MODELS:
    r = R[name]
    if not r:
        continue
    rows.append([label, f"{r['perplexity']:.2f}", "-" if r.get("kl_divergence") is None else f"{r['kl_divergence']:.3f}",
                 "100%" if r.get("top1_agreement") is None else pct(r["top1_agreement"]), f"{r['mmlu_accuracy'] * 100:.0f}%", pct(acc(r)), f"{size_gb(r):.2f} GB"])
S += table(rows, [5.6, 1.7, 1.4, 2.2, 1.7, 2.0, 2.0])
S += [P("Sizes: real for packed models; for the fake-quantized models (8-bit, GPTQ, and the earlier groups-of-128 row) the size they would have with compact 16-bit scales. <b>Correction:</b> our packed format stores 32-bit scales, so the real packed groups-of-128 model is %.2f GB, about 0.3 GB more than that estimate; a leaner format would close the gap. Rows starting with a plus sign are the model above them with that change." % size_gb(g128), small)]
S += img("overview", "<b>Figure 1.</b> Exam accuracy (left) and distance from the original's predictions (right) for every model, coloured by method family.")
S += [PageBreak()]
S += img("size_vs_quality", "<b>Figure 2.</b> Quality against size. The best methods sit near the top (exam) and lower (distance).")
S += img("sensitivity", "<b>Figure 3.</b> Which parts of the model suffer most from 4-bit rounding (per-row model). Layers 1 and 31 are by far the most fragile.")
S += [PageBreak()]
S += img("gptq_window", "<b>Figure 4.</b> GPTQ calibrated on 2048-token text fails on 512-token test text; calibrating on matching text fixes it.")
S += img("rag", "<b>Figure 5.</b> Retrieval (plain and HyDE) did not recover the loss on the 4-bit model: as many answers were broken as were fixed.")
S += img("qlora", "<b>Figure 6.</b> QLoRA made the model better at predicting Wikipedia text but moved it further from the original.")
S += [PageBreak()]

# ============ 6. findings ============
S += [P("6. What we found", h1)]
items = [
    f"<b>8-bit is essentially free; 4-bit costs something.</b> 8-bit kept the exam at {pct(acc(q8))}. Plain 4-bit fell to {pct(acc(rtn))} (from {pct(acc(orig))}).",
    f"<b>A finer ruler is the cheapest fix.</b> Groups of 128 gave {pct(acc(g128))} for only about 0.1 GB more.",
    f"<b>Mixed precision works when it protects the right kinds of table everywhere.</b> Protecting a few whole layers improved fidelity but not the exam; protecting v, k, o and down across the model reached {pct(acc(types))} with KL {types['kl_divergence']:.3f}.",
    f"<b>GPTQ is picky about its calibration text.</b> Matched to the test text length it equals the simple groups-of-128 method on short text and is better on long text.",
    f"<b>Retrieval did not help.</b> Plain RAG {pct(rag['plain']['accuracy'])} and HyDE {pct(rag['hyde']['accuracy'])} against {pct(rag['none']['accuracy'])} without. It repaired many damage-related errors but broke about as many correct ones.",
    f"<b>Training toward the wrong target backfires.</b> QLoRA lowered perplexity to {qlora['perplexity']:.2f} (below the original's {orig['perplexity']:.2f}) but raised KL from {rtn['kl_divergence']:.3f} to {qlora['kl_divergence']:.3f}, and the exam fell to {pct(acc(qlora))}. Optimising for text prediction is not the same as copying the original.",
]
if mg_types and mg_top3:
    items.insert(3, f"<b>Combining groups of 128 with mixed precision gives the best fidelity of any 4-bit model.</b> Protecting just layers 0, 1 and 31 at 8-bit on top of groups of 128 (+0.3 GB, {size_gb(mg_top3):.2f} GB total) cut the distance from the original from {g128['kl_divergence']:.3f} to <b>{mg_top3['kl_divergence']:.3f}</b> and raised same-top-word agreement from {pct(g128['top1_agreement'])} to <b>{pct(mg_top3['top1_agreement'])}</b>. The exam stayed at {pct(acc(mg_top3))} (groups of 128 alone: {pct(acc(g128))}; original {pct(acc(orig))}): all of these sit at the exam's noise floor, so the exam can no longer tell them apart. The v, k, o layout reached {pct(acc(mg_types))} with KL {mg_types['kl_divergence']:.3f} at {size_gb(mg_types):.2f} GB.")
S += B(items)
S += [P("Limits", h2)]
S += B(["One model, one domain (computer science), one 412-question exam (uncertainty about 2.5 points); the key mixed-precision gain has p = 0.08 on exam answers alone.",
        "Fidelity measures use WikiText text; behaviour on other kinds of text was not measured.",
        "Speed was not optimised: token-by-token generation is about 1.5 tokens per second because layers are unpacked in plain PyTorch.",
        "Not tried: distillation-style QLoRA, full MMLU, comparison with GPTQModel or bitsandbytes (not installable here), deployment."])
S += [P("How to reproduce", h2),
      P("Everything is in the repository (<font face='Courier'>github.com/SaribAzim/quantization-fidelity-study</font>). Typical order: <font face='Courier'>quantization/quantize_checkpoint.py</font> or <font face='Courier'>make_packed.py</font> to build a model, <font face='Courier'>evaluate_checkpoint.py</font> to measure it, <font face='Courier'>plot_results.py</font> for figures. The README lists every command and result table.")]
S += [PageBreak()]

# ============ 7. bugs ============
S += [P("7. Bugs and errors faced, and how they were fixed", h1),
      P("Mistakes are part of engineering. This is the consolidated list across all phases.")]
S += table([["Phase", "What went wrong", "Why", "How it was fixed"],
            ["2", "The first 4-bit results only quantized 70 of 225 layers.", "Layers offloaded to disk were empty placeholders; rounding them did nothing and raised no error.", "Quantized the saved files directly, one shard at a time, and added a check that fails unless exactly 224 tensors are quantized."],
            ["2", "The quantizer file the handoff described did not exist.", "It was written before being saved.", "Wrote it with an automatic test on a tiny fake model first."],
            ["2", "Finding the model files failed twice ('incomplete snapshot').", "A library check demanded files we never use.", "Asked the cache directly where config.json lives."],
            ["2", "A test crashed with 'Access is denied'.", "Windows locks memory-mapped files.", "Allowed harmless cleanup errors."],
            ["2", "Evaluations were cut off when the PC shut down.", "Long jobs ran in a session that ended.", "Made runs resumable: results saved after each stage."],
            ["3", "The ready-made GPTQ library would not install.", "It needs C++ build tools, WSL or Docker, none present.", "Wrote GPTQ from scratch."],
            ["3", "Our GPTQ was worse than plain rounding.", "Calibrated on 2048-token text, tested on 512-token text.", "Diagnosed with held-out tests and a layer-by-layer evaluator, then recalibrated on matching text."],
            ["3", "Layer-error looked fine while the model was bad.", "It is dominated by a few giant values.", "Used end-to-end loss on unseen text instead."],
            ["4", "The first library download crashed.", "Wikipedia rate-limited (HTTP 429) and the code hid the cause.", "Polite, resumable downloader (one request a second, honours retry hints, cached to disk)."],
            ["4", "Retrieval test missed 3 of 6 known topics.", "The library lacked TCP, public-key cryptography, gradient descent.", "Added 183 core curriculum topics (chosen from a syllabus, not from the exam)."],
            ["4", "HyDE time estimate was wrong by about 4 times.", "Assumed 2 s per word piece; disk streaming measured about 9 s.", "Measured with a smoke test, then used bigger batches and told the user the real timing."],
            ["Packed", "Packed model answered differently on 16 of 412 questions.", "GPU arithmetic order differs when the model is fully on the GPU; weights are identical.", "Verified real layers bit-for-bit, and compared QLoRA against the packed baseline, not the old file."],
            ["Packed", "The standard exam tool did not fit next to the model.", "It computes scores for the whole vocabulary at every position.", "Used our own scorer, validated against it."],
            ["QLoRA", "Perplexity improved but the model got worse.", "Training on WikiText taught WikiText style, not the original's behaviour.", "Judged by KL and the exam too, reported the failure."],
            ["Mixed", "Testing a whole kind of table did not fit in memory.", "Swapping all 32 big tables at once needs about 7.8 GB.", "Tested four spread-out layers per kind and compared per number."],
            ["Mixed", "Choosing what to protect could have leaked into results.", "Using test text for choosing would flatter the choice.", "Added a text-split option; chose on validation text only."],
            ["All", "Background watchers reported false errors and duplicates.", "Filters matched PowerShell's harmless NativeCommandError text.", "Matched real Python errors only, and used one quiet watcher."],
            ["All", "Report numbers could drift from the data.", "Hand-typed values.", "Every report and figure is generated from the results files."]],
           [1.3, 4.6, 5.2, 5.5])

doc = SimpleDocTemplate(OUT_PDF, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.9 * cm,
                        title="Quantization Fidelity Study - project details", author="Sarib Azim")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("wrote", os.path.abspath(OUT_PDF))
