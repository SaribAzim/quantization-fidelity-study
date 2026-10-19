"""
The models the website shows, in one place, shared by the site data exporter,
the gallery recorder and the live server.

Labels are written for the website: plain words and no hyphens or dashes
(the site's copy rule), so "4 bit" rather than "4-bit".

  runnable  fits the 8 GB GPU as a real packed model, so it can be recorded
            for the gallery and served live
  gallery   has recorded answers in the arena (the original runs offloaded,
            slowly, so it is recorded once and replayed)
"""

ORIGINAL_REPO = "meta-llama/Llama-3.1-8B-Instruct"

CATALOG = [
    dict(id="original", label="Original 16 bit", family="Original", result="baseline",
         runnable=False, gallery=True, featured=None,
         note="The reference model. 15 GB, too large for the 8 GB GPU, so its answers are recorded and replayed."),
    dict(id="g128_l0131", label="Groups of 128 + 8 bit layers 0, 1, 31", family="Mixed precision", result="packed_g128_mixed_top3",
         folder="quantized_models/packed_g128_mixed_top3", runnable=True, gallery=True, featured="Closest to the original",
         note="Finer rounding everywhere, with the three most fragile layers kept at 8 bit."),
    dict(id="per_row_vkod", label="Per row + 8 bit v, k, o, down", family="Mixed precision", result="packed_mixed_types",
         folder="quantized_models/packed_mixed_types", runnable=True, gallery=True, featured="Best exam score",
         note="Plain rounding, with the most fragile kinds of weight table kept at 8 bit in every layer."),
    dict(id="g128", label="Groups of 128", family="Naive, groups of 128", result="packed_naive_4bit_g128",
         folder="quantized_models/packed_naive_4bit_g128", runnable=True, gallery=True, featured="Smallest strong model",
         note="Every 128 neighbouring weights get their own rounding ruler."),
    dict(id="g128_vkoq", label="Groups of 128 + 8 bit v, k, o, q", family="Mixed precision", result="packed_g128_mixed_types",
         folder="quantized_models/packed_g128_mixed_types", runnable=True, gallery=True, featured=None,
         note="Finer rounding, with attention tables kept at 8 bit."),
    dict(id="per_row", label="Plain 4 bit, one ruler per row", family="Naive, one ruler per row", result="packed_naive_4bit",
         folder="quantized_models/packed_naive_4bit", runnable=True, gallery=True, featured=None,
         note="The simplest rounding. Shows the damage the other methods repair."),
    dict(id="qlora", label="Plain 4 bit + QLoRA", family="QLoRA", result="qlora_naive_4bit",
         folder="quantized_models/packed_naive_4bit", adapters="quantized_models/qlora/adapters_r16.pt", runnable=True, gallery=True, featured=None,
         note="Small trained adapters on top of plain 4 bit. Better at predicting text, further from the original."),
    # Results only (not runnable on 8 GB as real packed models, or not worth serving)
    dict(id="naive_8bit", label="Naive 8 bit", family="8 bit", result="naive_quant_8bit", runnable=False, gallery=False, featured=None,
         note="Almost lossless, but about 8.5 GB packed: just too large for the 8 GB GPU."),
    dict(id="per_row_l0131", label="Per row + 8 bit layers 0, 1, 31", family="Mixed precision", result="packed_mixed_top3", runnable=False, gallery=False, featured=None,
         note="Protecting three layers improves fidelity but not the exam."),
    dict(id="per_row_8layers", label="Per row + 8 bit for 8 layers", family="Mixed precision", result="packed_mixed_layers", runnable=False, gallery=False, featured=None,
         note="Protecting eight whole layers improves fidelity but not the exam."),
    dict(id="gptq_512", label="GPTQ, groups of 128", family="GPTQ", result="gptq_4bit_g128_cal512", runnable=False, gallery=False, featured=None,
         note="Calibrated rounding, written from scratch, calibrated on 512 token text."),
    dict(id="gptq_2048", label="GPTQ, long calibration text", family="GPTQ", result="gptq_4bit_g128", runnable=False, gallery=False, featured=None,
         note="The same method calibrated on 2048 token text: it fails on shorter text."),
]

BASELINE_ID = "per_row"  # the damaged starting point that "recovery" is measured from


def by_id(model_id):
    return next(m for m in CATALOG if m["id"] == model_id)
