"""
Builds study_guide.pdf: a teaching document for the project owner, covering the ideas, the work,
the results, likely interview questions and a self test. Numbers are read from the results files.

Run from the project root:
    python evaluation/make_study_guide.py
Writes ../study_guide.pdf (next to the project folder).
"""

import os
import sys

sys.path.insert(0, ".")

from PIL import Image as PI
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from evaluation.plot_results import load, size_gb

OUT_PDF = os.path.join("..", "study_guide.pdf")
FIG = "results/figures"
INK, INK2 = "#1d1a16", "#5c5448"
ACCENT = "#3d6b52"

ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=10.2, leading=15, textColor=colors.HexColor(INK), spaceAfter=6)
small = ParagraphStyle("small", parent=body, fontSize=8.5, leading=11.5, textColor=colors.HexColor(INK2))
h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=18, leading=22, spaceBefore=2, spaceAfter=8, textColor=colors.HexColor(ACCENT))
h2 = ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=12, leading=15, spaceBefore=9, spaceAfter=3)
title = ParagraphStyle("title", parent=h1, fontSize=32, leading=37, spaceAfter=4, textColor=colors.HexColor(INK))
bul = ParagraphStyle("bul", parent=body, leftIndent=14, bulletIndent=3, spaceAfter=3)
cell = ParagraphStyle("cell", parent=body, fontSize=8.6, leading=11.2, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")
qs = ParagraphStyle("q", parent=body, fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=2)


def P(t, s=body): return Paragraph(t, s)
def B(items): return [Paragraph(i, bul, bulletText="•") for i in items]
def code(t): return f"<font face='Courier' size='9'>{t}</font>"


def box(text, bg="#eef4ee", edge=ACCENT):
    t = Table([[Paragraph(text, body)]], colWidths=[16.6 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)), ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(edge)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return [t, Spacer(1, 7)]


def table(rows, widths):
    data = [[Paragraph(str(c), cellb if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[x * cm for x in widths], repeatRows=1)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d9d2c4")),
          ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9efe7"))]
    st += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f8f5ee")) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(st))
    return [t, Spacer(1, 8)]


def img(name, caption, width_cm=16.0):
    path = f"{FIG}/{name}.png"
    if not os.path.exists(path):
        return []
    pw, ph = PI.open(path).size
    return [KeepTogether([Image(path, width=width_cm * cm, height=width_cm * cm * ph / pw), P(caption, small)])]


def qa(q, a): return [P(q, qs), P(a)]


def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 8); c.setFillColor(colors.HexColor(INK2))
    c.drawString(2 * cm, 1.2 * cm, "Quantization Fidelity Study - study guide")
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {d.page}"); c.restoreState()


pct = lambda x: f"{x * 100:.1f}%"
acc = lambda r: r["mmlu_extended_cs"]["accuracy"]
R = {n: load(n) for n in ["baseline", "naive_quant_8bit", "packed_naive_4bit", "packed_naive_4bit_g128", "packed_mixed_types",
                          "packed_g128_mixed_top3", "gptq_4bit_g128_cal512", "qlora_naive_4bit"]}
orig, q8, rtn, g128, types, best, gptq, qlora = (R[k] for k in ["baseline", "naive_quant_8bit", "packed_naive_4bit", "packed_naive_4bit_g128",
                                                              "packed_mixed_types", "packed_g128_mixed_top3", "gptq_4bit_g128_cal512", "qlora_naive_4bit"])
rag = {m: load(f"rag/naive_quant_4bit_{m}") for m in ("none", "plain", "hyde")}
kl = lambda r: f"{r['kl_divergence']:.3f}"
top1 = lambda r: pct(r["top1_agreement"])

S = []

# ================= cover and how to use =================
S += [Spacer(1, 40), P("Quantization Fidelity Study", title),
      P("Study guide: understand it, explain it, defend it", ParagraphStyle("sub", parent=body, fontSize=14, textColor=colors.HexColor(INK2))),
      Spacer(1, 14),
      P("This guide is written for the project owner. It assumes no background beyond the project itself. Its job is that after reading it you can explain every part of the project to a recruiter or an engineer, answer follow up questions, and say honestly what the results do and do not show."),
      P("How to use it", h2)]
S += B(["<b>Part 1 to 3</b> give the big picture and the ideas, in plain words with small analogies. Read these first.",
        "<b>Part 4 and 5</b> cover what was built and what was found. <b>Part 6</b> is the website and deployment.",
        "<b>Part 7</b> lists the honest weaknesses. Interviewers respect people who know them.",
        "<b>Part 8</b> has interview questions with model answers, and <b>Part 9</b> is a self test with answers.",
        "<b>Part 10</b> is the glossary and <b>Part 11</b> the bugs we faced and how they were fixed."])
S += [Spacer(1, 10), P("The 30 second version", h2)]
S += box(f"An 8 billion parameter language model needs 15 GB, but the graphics card has 8 GB. Storing each number with 4 bits instead of 16 shrinks it about three times, but costs quality. This project measures that cost against the original model's own behaviour, and tests which fixes win it back. The best result is a <b>{size_gb(best):.1f} GB</b> model that scores <b>{pct(acc(best))}</b> against the original's <b>{pct(acc(orig))}</b> on a 412 question exam, and picks the same next word as the original <b>{top1(best)}</b> of the time. Retrieval and fine tuning did not help, and are reported as negative results. Everything is free, written from scratch, and live as a website.")
S += [PageBreak()]

# ================= Part 1 =================
S += [P("Part 1. The big picture", h1),
      P("What is a language model, really?", h2),
      P("A language model reads some text and predicts the next piece of text, over and over. Llama 3.1 8B Instruct does this using about <b>8 billion stored numbers</b>, called <b>weights</b>. The weights are the model's learned knowledge, arranged in big tables (matrices), one set per layer. The model has 32 layers, and each layer has several tables (called q, k, v, o, gate, up and down, which you do not need to memorise, but 'v, k, o and down' come up in the results)."),
      P("Why is size the problem?", h2),
      P("Each weight is normally stored with 16 bits of detail. Eight billion numbers times 16 bits is about <b>15 GB</b>. The project's graphics card, an RTX 3060 Ti, has <b>8 GB</b>. A model that does not fit on the card has to be streamed in from normal memory or disk, which makes it extremely slow: the original model here produced about 0.3 word pieces a second, against about 1.5 for the 4 bit model that fits."),
      P("What is quantization?", h2),
      P("Quantization means storing each weight with fewer bits. Think of saving a photograph at lower resolution: the picture is smaller and still recognisable, but some fine detail is gone. With 4 bits, each weight can only take one of 16 values instead of thousands. The model shrinks to about a quarter, but every weight is now slightly wrong, and thousands of small errors can add up."),
      P("What did this project actually ask?", h2)]
S += B(["How much quality is lost when the model is shrunk to 8 bit and to 4 bit?",
        "Not only 'what is the exam score', but <b>how faithfully does the small model copy the original's behaviour</b>? That is why the repository is called a fidelity study.",
        "Which techniques win the lost quality back, and which do not?",
        "Can the best version be shown to the world for free, without paying for a server?"])
S += [P("Why this is a good project to talk about", h2)]
S += B(["It is a real engineering problem (fit a big model on small hardware), not a tutorial.",
        "Every method was written from scratch, so you can explain how it works, not just call a library.",
        "It reports negative results honestly, which is what real research looks like.",
        "It is deployed, so anyone can see it."])
S += [PageBreak()]

# ================= Part 2 =================
S += [P("Part 2. The core ideas, one step at a time", h1),
      P("2.1 Naive rounding (round to nearest)", h2),
      P("Imagine a ruler with only 16 tick marks. For each weight, find the nearest tick, and store only which tick it was (a number from 0 to 15). That is 4 bits. To get the weight back later you need to know where the ruler starts and how far apart the ticks are, so each ruler stores two small numbers: a <b>scale</b> (spacing between ticks) and a <b>zero point</b> (where it starts)."),
      P("<b>Worked example.</b> Take a row whose weights run from -0.4 to +0.7. The ruler is stretched to cover that range with 16 ticks, so each gap is about 0.07. A weight of 0.03 is moved to the nearest tick, which might be 0.0 or 0.07. The error for that weight is up to half a gap, about 0.035. If most weights in the row are tiny, like 0.01, they all get squashed onto one or two ticks. That is the problem the next idea solves."),
      P("2.2 Why a few extreme weights hurt (the outlier problem)", h2),
      P("One unusually large weight stretches the ruler. The ticks spread further apart to cover it, and every ordinary weight in that row is rounded harshly. On the website's rounding explainer, 38 of the 40 sample weights land on the same single mark under 'one ruler per row'. This is the core reason plain 4 bit loses quality."),
      P("2.3 Groups of 128 (a ruler for every 128 weights)", h2),
      P("Instead of one ruler per whole row, give every group of 128 neighbouring weights its own small ruler. An extreme weight now only disturbs the 127 weights next to it. The ticks sit close to the weights they serve, so rounding is gentler. The cost is a few extra numbers to store: about 0.4 GB more across the model. On the sample row the average rounding error fell 4.3 times."),
      P("2.4 Real versus fake 4 bit storage", h2),
      P("Early on, the rounded weights were saved again as 16 bit numbers (called 'fake' quantization). That measures the quality loss correctly, but the file stays 15 GB, so it proves nothing about fitting the GPU. <b>Real packed storage</b> keeps the 4 bit tick numbers, two to a byte, plus the rulers, and unpacks one layer at a time just before it is used. The project proved the unpacked weights are bit for bit identical to the fake ones, so quality numbers are unaffected. The real size came out at 5.2 GB for per row 4 bit and 5.6 GB for groups of 128 (the estimate had been 5.3, a useful reminder to measure rather than trust estimates)."),
      P("2.5 GPTQ (rounding with a plan)", h2),
      P("Naive rounding treats every weight on its own. GPTQ is smarter. It shows the model a little sample text (the <b>calibration data</b>) to learn which inputs are large and which move together. Then it rounds the weights one column at a time, and after each column <b>adjusts the columns not yet rounded</b> to cancel out the error just made, like a carpenter who trims the next board to make up for one cut too short. The 'Hessian' mentioned in papers is just the record of which inputs matter and how they relate."),
      P("It was written by hand because the standard library could not be installed on this Windows setup. The important lesson: calibrated on long text (2048 pieces) it did badly on short test text (512 pieces), and the reverse. <b>Calibration data must look like the data the model will see</b>.")]
S += [P("2.6 Mixed precision (protect the fragile parts)", h2),
      P("Not every table suffers equally from rounding. The project measured this: take the 4 bit model, upgrade one group of weights to 8 bit, and see how much the model's output moves toward the original. Groups where the upgrade helped most are the <b>fragile</b> ones: layers 0, 1 and 31, and the v, k, o and down tables. Keeping just those at 8 bit costs well under 1 GB and recovers most of the loss. The choice was made on a separate validation text, never on the test text, so it cannot leak into the reported results."),
      P("2.7 RAG and HyDE (looking things up)", h2),
      P("Retrieval Augmented Generation gives the model a library to look things up in. The project built a library of 10,751 passages from Wikipedia computer science articles. For each question, find the most relevant passages and add them to the prompt. <b>HyDE</b> first lets the model write a pretend answer, then searches using that pretend answer instead of the raw question, because a draft answer often looks more like the real passages than the question does."),
      P("The hope was that giving the 4 bit model facts to read would make up for the facts it forgot. It did not (see Part 5).")]
S += [P("2.8 QLoRA (training a small correction)", h2),
      P("Keep the 4 bit weights frozen and train a few small extra tables (adapters) on top. It learns from WikiText. The adapters did make the model better at Wikipedia style text (its perplexity fell), but it learned the training text's quirks, such as putting a space before commas, and moved <b>further from the original's behaviour</b>. For a fidelity study, that is the wrong direction.")]
S += [PageBreak()]

# ================= Part 3 =================
S += [P("Part 3. How quality is measured", h1),
      P("Four measures, each answering a different question. Know these well, because choosing the right yardstick is the heart of the project.", body)]
S += table([["Measure", "What it asks", "How to read it", "Strength and weakness"],
            ["Exam score (MMLU, 4 computer science subjects, 412 questions)", "Does the model still answer multiple choice questions correctly?", "Higher is better. The original scores 55.6%.", "Easy to understand, but only 412 questions: about 2.5 points of random noise either way, so small gaps mean nothing."],
            ["Perplexity (on WikiText)", "How surprised is the model by ordinary text?", "Lower is better. The original is 11.44.", "Smooth, but a model can improve it by learning the test text's style (QLoRA did this), so it is not about fidelity."],
            ["Distance from the original (called KL)", "How different are the model's probabilities from the original's, word by word?", "0 means identical. Plain 4 bit is 0.281, the best 4 bit is 0.049.", "Built on thousands of predictions, so it is the most sensitive and the key fidelity measure."],
            ["Same top word (top 1 agreement)", "How often does the model pick the same most likely next word as the original?", "100% means identical. Plain 4 bit 74.3%, best 89.8%.", "Easy to explain: 'it agrees with the original 9 times in 10'."]],
           [3.6, 4.0, 4.2, 4.8])
S += box("<b>Why the exam alone is not enough.</b> The exam has plus or minus 2.5 points of noise. Several 4 bit models score within that noise of the original, so the exam cannot tell them apart. The distance and same-top-word measures use thousands of predictions and can. A good answer in an interview is: 'I used the exam because people recognise it, but I leaned on distance from the original because it is statistically stronger.'")
S += [P("Fairness rules followed", h2)]
S += B(["Every model was measured with the same inputs and the same code.",
        "Choices (which layers to protect, which settings) were made on validation text. Reported numbers come from test text and the exam, which were never used for choosing.",
        "Answers were compared question by question, counting how many were fixed and how many broken, not only totals. A sign test says whether a gain is more than luck.",
        "Settings were fixed before looking at exam results."])
S += [PageBreak()]

# ================= Part 4 =================
S += [P("Part 4. What was built, in order", h1)]
S += table([["Step", "What was done", "Result"],
            ["Setup and baseline", "Python 3.11, PyTorch on the 3060 Ti. The model was too big for the card, so it ran with parts offloaded to memory and disk. Measured once as the reference every other model is compared with.", f"Original: exam {pct(acc(orig))}, perplexity 11.44, 15 GB."],
            ["Naive 8 bit and 4 bit", "Wrote round to nearest quantization from scratch and applied it to all 224 weight tables, layer by layer from the saved files so the whole model never has to fit in memory.", f"8 bit: {pct(acc(q8))}, almost lossless but about 8.5 GB. Plain 4 bit: {pct(acc(rtn))}."],
            ["Groups of 128 and GPTQ", "Added the finer rulers and a hand written GPTQ.", f"Groups of 128: {pct(acc(g128))}. GPTQ: {pct(acc(gptq))} (needs matching calibration)."],
            ["Real 4 bit storage", "Packed two 4 bit numbers per byte, with unpacking per layer.", f"{size_gb(rtn):.1f} GB, fits the GPU, about 5 times faster than offloading."],
            ["Retrieval", "Built the 10,751 passage library and the plain and HyDE pipelines.", "No recovery (Part 5)."],
            ["QLoRA", "Trained adapters on the frozen 4 bit model.", f"Exam {pct(acc(qlora))}, further from the original."],
            ["Mixed precision", "Measured sensitivity, kept fragile parts at 8 bit, tested three layouts.", f"Best per row layout: {pct(acc(types))}, distance {kl(types)}."],
            ["Combination", "Groups of 128 plus mixed precision, sensitivity re measured on the finer model.", f"Best model: {size_gb(best):.2f} GB, exam {pct(acc(best))}, distance {kl(best)}, same top word {top1(best)}."],
            ["Website and deployment", "Three page site, local live backend, GitHub Pages, MIT licence.", "Live and free (Part 6)."]],
           [3.2, 8.2, 5.2])
S += img("pipeline", "The pipeline: original model, quantize, pack to real 4 bit, repair, measure.", 12.5)
S += [Spacer(1, 6), P("How a 15 GB model was handled on an 8 GB card", h2),
      P("The model is stored as several files. The code reads one layer's weights at a time, rounds or measures them, and writes them out before moving on. Peak memory is one layer, not the whole model. This 'layer by layer' pipeline is also why the project can quantize and evaluate a model that does not fit on the card."),
      P("The code map", h2)]
S += table([["Folder or file", "What it holds"],
            [code("quantization/naive_ptq.py"), "Round to nearest, per row and per group"],
            [code("quantization/gptq.py, run_gptq.py"), "The hand written GPTQ"],
            [code("quantization/packed.py"), "Real 4 bit storage and unpacking"],
            [code("quantization/mixed_sensitivity.py"), "Which parts are fragile"],
            [code("evaluation/"), "Exam, perplexity, distance measures and the plots"],
            [code("rag/"), "The retrieval library, plain RAG and HyDE"],
            [code("deployment/"), "The local live server, the recorded answers builder, site data export"],
            [code("web/"), "The website (React, TypeScript, Vite)"],
            [code("results/"), "Every measured result as a data file, plus figures"]],
           [6.4, 10.2])
S += [Spacer(1, 14)]

# ================= Part 5 =================
S += [P("Part 5. What was found", h1)]
rows = [["Model", "Exam", "Distance (KL)", "Same top word", "Size"]]
for label, r in [("Original, 16 bit", orig), ("Naive 8 bit", q8), ("Plain 4 bit, one ruler per row", rtn), ("4 bit, groups of 128", g128),
                 ("Per row + 8 bit for fragile tables (v,k,o,down)", types), ("GPTQ, groups of 128", gptq),
                 ("Groups of 128 + 8 bit layers 0, 1, 31 (best)", best), ("Plain 4 bit + QLoRA", qlora)]:
    rows.append([label, pct(acc(r)), "reference" if r.get("kl_divergence") is None else kl(r),
                 "reference" if r.get("top1_agreement") is None else top1(r), f"{size_gb(r):.2f} GB"])
S += table(rows, [7.2, 1.9, 2.6, 2.6, 2.3])
S += img("recovery", "How much of plain 4 bit's loss each technique wins back. Green recovers, red makes it worse.")
S += [P("What the results mean", h2)]
S += B([f"<b>8 bit is nearly free quality wise</b> ({pct(acc(q8))} against {pct(acc(orig))}), but at 8.5 GB it still does not fit the 8 GB card. So 8 bit is not the answer here; 4 bit is the only size that fits.",
        f"<b>Plain 4 bit costs real quality.</b> The exam drops about 5 points and the model disagrees with the original on about one word in four.",
        "<b>Finer rulers (groups of 128) and protecting fragile parts both help, and together help most.</b> The best model closes most of the gap at 5.9 GB.",
        "<b>The exam cannot separate the best 4 bit models</b> from each other or from the original, because they all sit within its noise. The distance measure can.",
        "<b>GPTQ was not a clear win.</b> Done carefully it matched or slightly beat naive rounding, but it is sensitive to calibration text, and the simple fixes (groups, mixed precision) did as well or better here."])
S += [P("The negative results (just as important)", h2)]
S += table([["Technique", "Result", "Why it probably failed"],
            ["Plain retrieval (RAG)", f"Exam {pct(rag['plain']['accuracy']) if rag['plain'] else 'n/a'} against {pct(rag['none']['accuracy']) if rag['none'] else 'n/a'} without it. It fixed 53 answers and broke 61.", "Retrieved passages often distract the model. The quality loss from rounding is not missing facts, it is slightly corrupted reasoning, and a library cannot repair that."],
            ["HyDE", "Same score as no retrieval: 53 fixed, 53 broken.", "A better search does not matter if the problem is not missing knowledge."],
            ["QLoRA", f"Perplexity improved (9.63) but the exam fell to {pct(acc(qlora))} and the distance from the original rose to {kl(qlora)}.", "It learned WikiText's style (such as a space before commas) instead of recovering the original's behaviour. Training toward the original's own outputs (distillation) was not done."]],
           [3.2, 6.2, 7.2])
S += img("size_vs_quality", "Size against quality: the best 4 bit models sit left of the 8 GB line and near the original's score.")
S += [PageBreak()]

# ================= Part 6 =================
S += [P("Part 6. The website and deployment", h1),
      P("What it is", h2),
      P("A free static website in <font face='Courier'>web/</font>, built with React, TypeScript and Vite, hosted on GitHub Pages, with three pages:"),
      ]
S += B(["<b>The project:</b> a scroll driven hero where your scroll quantizes a real grid of weights from layer 1 (16 bit to 8 bit to 4 bit, with the title narrowing and the size counting down past the 8 GB line), a rounding explainer, the results table and charts, what did not help, and how it was built.",
        "<b>Model answers:</b> the original, 8 bit and best 4 bit answering the same ten questions, replayed word by word. Differences from the original are highlighted, and a strip under each answer shows where it first drifts. A chart shows how much of each answer's wording matches the original across all ten questions (8 bit averages 77%, best 4 bit 49%).",
        "<b>Workbench:</b> your own testing page. Any model's recorded answers can be replayed, and with the local backend running, the 4 bit models answer new questions live."])
S += [P("Why it works for free", h2),
      P("A language model cannot run on GitHub Pages. So the site is static: every answer was <b>recorded in advance</b> on the project's graphics card (including the slow 8 bit and original models, which ran partly from memory over about an hour) and is replayed in the browser at real speed or faster. Visitors never use your graphics card."),
      P("How live testing works", h2),
      P("A small server (<font face='Courier'>deployment/server.py</font>, FastAPI) runs on your PC at 127.0.0.1:8000 and streams tokens to the page as they are generated. It only accepts connections from your own computer, so nobody else can reach it. On the public site the page never contacts it until you press <b>Connect to a local backend</b>, so visitors are not prompted for local network access. Start it with <font face='Courier'>deployment/run_live.ps1</font>, stop it with Ctrl+C."),
      P("How deployment works", h2)]
S += B(["A GitHub Actions workflow (<font face='Courier'>.github/workflows/pages.yml</font>) builds the site and publishes it on every push to the main branch.",
        "Before the repository was made public, the entire git history was scanned for secrets, weights, large files and personal paths. It was clean.",
        "The repository carries the MIT licence for the code. Llama 3.1 and its answers remain under Meta's licence, and no model weights are hosted anywhere.",
        "Live address: https://saribazim.github.io/quantization-fidelity-study/"])
S += [P("Design decisions worth mentioning", h2)]
S += B(["Cream and ivory theme chosen by the owner, with ink blue for the original, sage for the best 4 bit model and mustard for the 8 GB line.",
        "No hyphens or dashes in any site text, enforced by an automated check that also screenshots every page at desktop, tablet and phone widths and in dark mode.",
        "Reduced motion is respected: the hero is not pinned and charts show final states."])
S += [PageBreak()]

# ================= Part 7 =================
S += [P("Part 7. The honest limits", h1),
      P("Knowing these is a strength in an interview. Say them before you are asked.", body)]
S += B(["<b>One model, one domain.</b> Only Llama 3.1 8B, and the exam covers four computer science subjects (412 questions), not the full 57 subject MMLU.",
        "<b>The exam is noisy.</b> About plus or minus 2.5 points. The headline gain from mixed precision has p = 0.08 on exam answers alone, which is suggestive, not conclusive. The distance measure is the stronger evidence.",
        "<b>Fidelity was measured on WikiText.</b> Behaviour on other kinds of text was not measured.",
        "<b>Speed is slow</b> (about 1.5 word pieces a second) because 4 bit layers are unpacked in plain PyTorch. Fast 4 bit needs compiled kernels, which could not be built on this Windows setup. A real deployment would use them.",
        "<b>Not done:</b> distillation style QLoRA (training toward the original's outputs), comparison with GPTQModel and bitsandbytes, and the full 57 subject exam.",
        "<b>The website replays recordings.</b> Visitors cannot ask new questions; only the owner can, on a local GPU.",
        "<b>The recorded answer comparison</b> uses only ten questions, so it illustrates behaviour rather than proving it."])
S += [P("What you would do next", h2)]
S += B(["Distillation QLoRA: train the adapters to match the original model's outputs instead of WikiText.",
        "Run the full 57 subject exam for a tighter statistical comparison.",
        "Compare with bitsandbytes and GPTQModel on Linux, with fast kernels, and measure real speed.",
        "Try other models to see whether the fragile layers (0, 1 and 31) are the same."])
S += [PageBreak()]

# ================= Part 8 =================
S += [P("Part 8. Interview questions with model answers", h1)]
S += qa("Describe the project in one minute.",
        f"Llama 3.1 8B needs 15 GB and my GPU has 8, so I quantized it to 4 bit and measured how faithfully the small model copies the original, not just its exam score. I wrote naive rounding, GPTQ and real packed 4 bit storage from scratch. Plain 4 bit drops the exam from {pct(acc(orig))} to {pct(acc(rtn))}. Finer groups of 128 weights plus keeping the most fragile layers at 8 bit gets a {size_gb(best):.1f} GB model to {pct(acc(best))} that agrees with the original's top word {top1(best)} of the time. Retrieval and fine tuning did not help. It is live as a free website.")
S += qa("Why did you not just use bitsandbytes or a library?",
        "I wanted to understand and control the mechanics, and to measure each choice cleanly. The GPTQ library also could not be installed on Windows. The cost was speed: my packed layers are unpacked in plain PyTorch, so generation is slow. A production system would use compiled kernels.")
S += qa("What is the difference between naive quantization and GPTQ?",
        "Naive rounding moves each weight to its nearest allowed value independently. GPTQ uses sample text to learn which inputs matter, rounds column by column, and adjusts the remaining columns to cancel the error. It is more accurate in principle but sensitive to the calibration text, which I found when 2048 token calibration collapsed on 512 token test text.")
S += qa("Why are groups of 128 better than one ruler per row?",
        "One extreme weight stretches a row's ruler, so every ordinary weight in that row is rounded harshly. With a ruler per 128 weights, an outlier only affects its neighbours. On a sample row the rounding error fell 4.3 times, at about 0.4 GB extra storage.")
S += qa("Why did you measure distance from the original and not just accuracy?",
        "The exam has 412 questions and about 2.5 points of noise, so several models were statistically tied. The distance and same top word measures use thousands of predictions and separate them. And the study's question is fidelity, how closely the small model copies the original.")
S += qa("How did you choose which layers to keep at 8 bit?",
        "By sensitivity: I upgraded one group at a time in the 4 bit model and measured how much closer the output moved to the original. The choice used validation text only, never the test text or exam, so it cannot inflate the reported numbers.")
S += qa("Why did retrieval not help?",
        "The quality loss from rounding is slightly corrupted reasoning, not missing facts. Giving the model passages to read fixed 53 questions but broke 61, so retrieval added as much noise as signal. HyDE fixed 53 and broke 53. A library cannot repair corrupted weights.")
S += qa("QLoRA improved perplexity. Why call it a failure?",
        "Because it improved by learning WikiText's style, for example spacing before commas, and moved further from the original: distance rose from 0.281 to 0.445 and the exam fell. For a fidelity study that is the wrong direction. Training toward the original's own outputs would be the better test, and is the next step.")
S += qa("How is the website free if it shows a model answering?",
        "The answers were recorded on my GPU ahead of time and are replayed in the browser, so GitHub Pages only serves static files. Visitors never use my GPU. Live questions work only when I run a local backend on my own machine, which only accepts local connections.")
S += qa("What was the hardest bug?",
        "My first 4 bit results quantized only 70 of 225 layers. The model does not fit in memory, so some layers were parked on disk as empty placeholders, and rounding an empty placeholder does nothing and raises no error. I fixed it by quantizing the saved weight files directly, one file at a time, and adding a check that fails unless exactly 224 tensors are quantized. The lesson was to count what you changed, not just trust that the code ran.")
S += qa("What would you change if you did it again?",
        "Run the full 57 subject exam from the start, use compiled 4 bit kernels so speed is realistic, and try distillation based QLoRA. I would also record every exam answer per question from the very start, because comparing fixed against broken answers was the most informative analysis.")
S += [PageBreak()]

# ================= Part 9 =================
S += [P("Part 9. Self test", h1),
      P("Cover the answers, try each question out loud, then check.", body)]
quiz = [
    ("How big is the original model, and why is that a problem?", "About 15 GB in 16 bit. The GPU has 8 GB, so it does not fit and runs very slowly from memory and disk."),
    ("What does 4 bit mean for a weight?", "It can take only one of 16 values, stored as a tick number from 0 to 15, with a scale and zero point to turn it back into a number."),
    ("Why does one huge weight in a row hurt?", "It stretches the ruler, so the ticks spread apart and ordinary weights are rounded harshly."),
    ("What is the fix, and what does it cost?", "A ruler per group of 128 weights. About 0.4 GB more storage."),
    ("What is the difference between fake and real 4 bit?", "Fake keeps rounded values in 16 bit storage (correct quality, same size). Real packs two 4 bit numbers per byte and shrinks the model to about 5 GB."),
    ("Which parts of the model are most fragile?", "Layers 0, 1 and 31, and the v, k, o and down tables."),
    ("What are the four quality measures?", "Exam score, perplexity, distance from the original (KL), same top word."),
    ("Which measure is the most sensitive, and why?", "Distance from the original, because it uses thousands of predictions instead of 412 questions."),
    ("What is the best model, and what are its numbers?", f"Groups of 128 with 8 bit layers 0, 1 and 31: {size_gb(best):.1f} GB, exam {pct(acc(best))}, distance {kl(best)}, same top word {top1(best)}."),
    ("Does 8 bit solve the problem?", "Not on this card: it is almost lossless but about 8.5 GB, too big for 8 GB."),
    ("What does HyDE do?", "It lets the model write a draft answer first, then searches for passages using the draft instead of the question."),
    ("Why did retrieval not recover the loss?", "The loss is corrupted reasoning, not missing facts, so extra passages added as much noise as help."),
    ("What did QLoRA learn that was unhelpful?", "WikiText's style, such as a space before commas, so it moved further from the original."),
    ("How does the live site avoid using your GPU?", "Answers are pre recorded and replayed. Only a local backend on your own PC can generate new ones."),
    ("Why is the exam alone not enough?", "About 2.5 points of noise on 412 questions, so close models are statistically tied."),
    ("Name two things that were not done.", "Distillation style QLoRA, the full 57 subject exam, comparison with bitsandbytes and GPTQModel, fast compiled kernels."),
]
for i, (q, a) in enumerate(quiz, 1):
    S += [P(f"{i}. {q}", qs), P(f"<font color='{INK2}'>Answer: {a}</font>", body)]
S += [PageBreak()]

# ================= Part 10 =================
S += [P("Part 10. Glossary", h1)]
S += table([["Term", "Plain meaning"],
            ["Weight", "One of the model's 8 billion stored numbers; its learned knowledge."],
            ["Layer", "One stage of the model. Llama 3.1 8B has 32, numbered 0 to 31."],
            ["Quantization", "Storing each weight with fewer bits to make the model smaller."],
            ["Bit", "A single yes or no digit. 4 bits give 16 possible values; 8 bits give 256; 16 bits give thousands."],
            ["Scale and zero point", "The two numbers that define a ruler: tick spacing and starting point."],
            ["Per row / group of 128", "How many weights share one ruler: a whole row, or 128 neighbours."],
            ["Round to nearest (naive, RTN)", "Moving each weight to the nearest allowed value on its ruler."],
            ["GPTQ", "Rounding column by column with error correction, guided by calibration text."],
            ["Calibration data", "A small sample of text used to guide GPTQ. It should resemble the real use."],
            ["Packed storage", "Two 4 bit numbers kept in one byte, unpacked a layer at a time."],
            ["Mixed precision", "Keeping fragile parts at 8 bit while the rest is 4 bit."],
            ["Perplexity", "How surprised the model is by text. Lower is better."],
            ["KL divergence (distance)", "How far the model's probabilities are from the original's. 0 is identical."],
            ["Top 1 agreement", "How often a model picks the same most likely next word as the original."],
            ["MMLU", "A multiple choice exam. Here: 4 computer science subjects, 412 questions."],
            ["RAG", "Looking passages up and adding them to the prompt before answering."],
            ["HyDE", "RAG that searches using a draft answer instead of the raw question."],
            ["QLoRA", "Training small adapter tables on top of a frozen 4 bit model."],
            ["Offloading", "Keeping parts of a too big model in RAM or on disk and moving them in as needed."],
            ["Word piece (token)", "The chunks of text a model reads and writes; often a word or part of a word."],
            ["GitHub Pages", "Free hosting for static websites from a GitHub repository."]],
           [4.8, 11.8])
S += [PageBreak()]

# ================= Part 11 =================
S += [P("Part 11. Bugs and errors faced, and how they were fixed", h1),
      P("Honest accounts of what went wrong are useful interview material. Each one shows a way of finding and fixing a problem.", body)]
S += table([["What went wrong", "Why", "How it was fixed"],
            ['The first 4 bit results only quantized 70 of 225 layers.', 'Layers parked on disk were empty placeholders, so rounding them did nothing and raised no error.', 'Quantized the saved weight files directly, one file at a time, and added a check that fails unless exactly 224 tensors are quantized.'],
            ["Finding the model files failed twice ('incomplete snapshot').", 'A library check demanded files we never use.', 'Asked the cache directly where the config file lives.'],
            ["A test crashed with 'Access is denied' on Windows.", 'Windows locks memory mapped files.', 'Allowed the harmless cleanup errors.'],
            ['Evaluations were cut off when the PC shut down.', 'Long jobs ran in a session that ended.', 'Made runs resumable: results are saved after each stage.'],
            ['The ready made GPTQ library would not install.', 'It needs C++ build tools, WSL or Docker, and none were present.', 'Wrote GPTQ from scratch, which also made the mechanics clear.'],
            ['Our GPTQ was worse than plain rounding.', 'Calibrated on 2048 piece text but tested on 512 piece text.', 'Diagnosed with held out tests and a layer by layer evaluator, then recalibrated on matching text.'],
            ['Layer error looked fine while the model was bad.', 'It is dominated by a few giant values.', 'Used end to end loss on unseen text instead.'],
            ['The first Wikipedia download crashed.', 'Wikipedia rate limited the requests (error 429) and the code hid the cause.', 'A polite, resumable downloader: one request a second, honours retry hints, cached to disk.'],
            ['The retrieval test missed 3 of 6 known topics.', 'The library lacked TCP, public key cryptography and gradient descent.', 'Added 183 core curriculum topics, chosen from a syllabus and not from the exam.'],
            ['The HyDE time estimate was wrong by about 4 times.', 'Assumed 2 seconds per word piece, but disk streaming measured about 9.', 'Measured with a smoke test, then used bigger batches and reported the real timing.'],
            ['The packed model answered 16 of 412 questions differently.', 'GPU arithmetic order differs when the model is fully on the GPU; the weights were identical.', 'Verified real layers bit for bit, and compared QLoRA against the packed baseline.'],
            ['The standard exam tool did not fit next to the model.', 'It computes scores for the whole vocabulary at every position.', 'Used our own scorer, validated against it.'],
            ['QLoRA improved perplexity but the model got worse.', "Training on WikiText taught WikiText style, not the original's behaviour.", 'Judged it by distance and the exam too, and reported the failure.'],
            ['Testing a whole kind of table at 8 bit did not fit in memory.', 'Swapping all 32 big tables at once needs about 7.8 GB.', 'Tested four spread out layers per kind and compared per number.'],
            ['Choosing what to protect could have leaked into results.', 'Using test text for choosing would flatter the choice.', 'Added a text split option and chose on validation text only.'],
            ['Background watchers reported false errors and duplicates.', "Filters matched PowerShell's harmless NativeCommandError text.", 'Matched real Python errors only, and used one quiet watcher.'],
            ['The packed size estimate (5.3 GB) differed from the real size (5.6 GB).', 'Estimates ignore the extra ruler numbers.', 'Reported measured sizes and marked estimates with an asterisk.'],
            ['Chrome blocked the website from reaching the local backend (error 400).', "Chrome's private network preflight check was not answered.", 'Enabled the private network header in the server, and added a test.'],
            ['Website: a TypeScript error in the tester page.', 'A check compared a state that could never match.', 'Removed the impossible check and rebuilt.'],
            ["Website: the hero had no side margin on phones, and the title read 'Llama 3.18B' when narrow.", 'A spacing rule cancelled the page margin, and the narrow letters hid the gap between words.', 'Changed it to vertical padding only and added word spacing.'],
            ['Website: 38 of 40 sample weights hid on top of each other in the rounding figure.', 'They really do land on the same mark, so they overlapped.', 'Spread overlapping dots into rows and labelled how many land on each mark.'],
            ['Recording the 8 bit model looked stalled and the site showed no answers.', 'The model does not fit the card, so it ran at 0.1 to 0.3 word pieces a second, and the site data is only updated at the end.', 'Waited for the full recording, then refreshed the site data and rebuilt.'],
            ['The GitHub command line tool was not installed.', 'Not present on this PC.', "Checked publish runs through GitHub's public API once the repository was public."],
            ['Report numbers could drift from the data.', 'Hand typed values.', 'Every report and figure is generated from the results files.'],
            ],
           [5.4, 5.2, 6.0])
S += [Spacer(1, 8), P("You now have the whole project in one document. If one part feels weak, go back to the matching Part, say it out loud in your own words, and check against the answer.", small)]

doc = SimpleDocTemplate(OUT_PDF, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.9 * cm,
                        title="Quantization Fidelity Study: study guide", author="Sarib Azim")
doc.build(S, onFirstPage=lambda c, d: None, onLaterPages=footer)
print(f"Wrote {OUT_PDF}")
