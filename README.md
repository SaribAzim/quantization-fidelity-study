# Quantization Fidelity Study

**Research question:** How much does an 8B parameter LLM degrade when quantized, and can a RAG architecture recover the fidelity lost in a specific domain?

## Model

Llama 3.1 8B Instruct (Meta, via Hugging Face)

## Project structure

```
quantization/   - Our own PTQ and GPTQ-style quantization implementations
evaluation/     - Evaluation harness (perplexity, task accuracy, latency, memory)
rag/            - RAG pipeline built on the quantized model
deployment/     - FastAPI service + Docker setup
results/        - Logged metrics and comparison tables from every phase
notebooks/      - Exploratory work, not the source of truth for final code
```

## Evaluation metrics

Every phase (base model, naive-quantized, GPTQ-quantized, RAG-augmented) is measured on the same four metrics, so comparisons stay apples-to-apples:

- **Perplexity** — language modeling quality on held-out text (WikiText-2, our own implementation in `evaluate.py`)
- **Task accuracy** — MMLU `high_school_computer_science` subtask, 0-shot, 50 samples, via `lm-evaluation-harness`:
  ```
  lm_eval --model hf --model_args pretrained=<model>,dtype=bfloat16,parallelize=True,offload_folder=offload --tasks mmlu_high_school_computer_science --apply_chat_template --limit 50
  ```
  (We initially hand-rolled this, but raw logit comparison proved too fragile for an instruction-tuned model's chat template and tokenizer quirks, `lm-eval` handles this correctly and is the standard tool for the job.)
- **Latency** — tokens per second
- **Memory footprint** — peak VRAM usage and model size on disk

## Status

Project in progress. Phases 0-2 done (baseline, naive quantization of all 224 decoder Linear layers).

### Phase 2 results (naive per-channel round-to-nearest, fake-quantized)

| | Base BF16 | Naive 8-bit | Naive 4-bit |
|---|---|---|---|
| Perplexity (WikiText-2) | 11.443 | 11.463 | 14.709 |
| KL divergence vs base | - | 0.0016 | 0.280 |
| Top-1 agreement with base | - | 97.8% | 74.3% |
| MMLU CS, first 50 q (stderr ~0.07) | 0.66 | 0.66 | 0.58 |
| MMLU CS extended, 412 q (stderr ~0.025) | 0.556 | 0.558 | 0.512 |
| Size if packed (GB, estimate) | 14.96 | 8.46 | 5.21 |

### Phase 3 results (GPTQ written from scratch, 4-bit, group size 128)

`gptqmodel` does not build on this Windows setup, so GPTQ is implemented in `quantization/gptq.py` and run layer by layer (`quantization/run_gptq.py`, fits 8 GB VRAM). All models below use 4 bits.

| | Naive per-row | Naive g128 | GPTQ g128 (calibrated on 2048-token text) | GPTQ g128 (calibrated on 512-token text) |
|---|---|---|---|---|
| Perplexity, 512-token windows (base 11.443) | 14.709 | 12.291 | 19.451 | 12.721 |
| KL divergence vs base | 0.280 | 0.085 | 0.543 | 0.100 |
| Top-1 agreement with base | 74.3% | 86.5% | 80.3% | 88.4% |
| MMLU CS, 412 q (base 0.556) | 0.512 | 0.546 | 0.534 | 0.546 |
| Perplexity, 2048-token windows (base 8.053) | - | 8.601 | 8.392 | 8.458 |

Findings: group-wise rounding helps naive quantization a lot. GPTQ overfits to the window length of its calibration text (2048-token calibration collapses on 512-token test windows), so calibration text must match real use. After recalibrating on 512-token windows, GPTQ and naive g128 are about equal on short text (identical exam score) and GPTQ is better on long text. Diagnostics are in `quantization/diagnose_gptq*.py` and `quantization/chain_eval.py`.

### Phase 4 results (RAG and HyDE on the naive 4-bit per-row model)

Library: 10,751 passages from about 780 Wikipedia computer-science articles (`rag/build_corpus.py`, contamination-checked against the exam), embedded with `BAAI/bge-small-en-v1.5` and searched by a plain matrix product (`rag/`). Same 412 CS exam questions, same naive 4-bit model, top-3 passages. The scorer (`rag/scorer.py`) matches lm-eval question by question on 404/412 questions with retrieval off.

| Naive 4-bit (per row), 412 questions | Accuracy | Fixed / broken vs no RAG | Sign test |
|---|---|---|---|
| Original model, no RAG (target) | 55.6% | - | - |
| No RAG | 51.2% | - | - |
| Plain RAG | 49.3% | 53 / 61 | p = 0.51 |
| HyDE RAG | 51.2% | 53 / 53 | p = 1.00 |

Neither retrieval mode recovered any of the 4.4-point loss on this exam. Reasons are discussed (untested) in `progress3.pdf`. Run: `python rag/run_rag_eval.py --model quantized_models/naive_4bit --name naive_quant_4bit --modes none hyde_gen plain hyde`.

### Real 4-bit storage and QLoRA (naive 4-bit per row)

`quantization/make_packed.py` stores the RTN 4-bit model as real packed integers (two per byte, plus per-row scale and zero-point) instead of 16-bit fake values: **5.22 GB instead of 14.96 GB**, fully on the 8 GB GPU (peak 5.89 GB). Unpacked weights are bit-identical to the fake-quantized ones (`test_packed.py`, plus a check on real layers), so results match the earlier model up to floating-point noise.

QLoRA (`quantization/train_qlora.py`): frozen packed 4-bit model, 41.9 M trainable LoRA parameters (rank 16), trained on WikiText-2 train windows of 512 tokens, 1 M tokens, 74 minutes, peak 6.06 GB VRAM. Fixed settings chosen before looking at the exam.

| | Original | RTN 4-bit (packed) | RTN 4-bit + QLoRA |
|---|---|---|---|
| Perplexity (WikiText-2) | 11.44 | 14.70 | **9.63** |
| KL divergence from original | - | 0.281 | 0.445 |
| Same top word as original | 100% | 74.3% | 77.3% |
| CS exam, 412 questions | 55.6% | 50.5% | 48.5% |

QLoRA fixed 47 exam answers and broke 55 (sign test p = 0.49): no recovery. Perplexity fell below the original model's because the adapters learned WikiText's style, while fidelity to the original (KL) got worse: next-word prediction on raw text is not the same objective as matching the original model. Distillation from the original model is the natural next experiment. Details in `progress4.pdf`.

### Mixed precision (naive 4-bit, fragile parts kept at 8-bit)

`quantization/mixed_sensitivity.py` swaps one group of layers at a time to 8-bit in the real packed 4-bit model and measures how much the output changes (WikiText *validation* text, so the choice cannot leak into reported results). Layers 0, 1 and 31 are by far the most sensitive; by kind of table, `down`, `up`, and (per number) `v` and `k`. `quantization/plan_mixed.py` builds layouts with equal extra budgets; `make_packed.py --plan` builds them as real packed models.

| 4-bit model | Extra size | Total size | KL from original | Same top word | Perplexity | CS exam, 412 q |
|---|---|---|---|---|---|---|
| Original | - | 15 GB | - | 100% | 11.44 | 55.6% |
| RTN 4-bit (packed) | - | 5.22 GB | 0.281 | 74.3% | 14.70 | 50.5% |
| + layers 0, 1, 31 at 8-bit | +0.30 GB | 5.52 GB | 0.132 | 83.6% | 13.03 | 48.3% |
| + 8 most sensitive layers at 8-bit | +0.81 GB | 6.03 GB | 0.101 | 84.4% | 12.67 | 49.5% |
| **+ v, k, o everywhere and down in 19 layers at 8-bit** | +0.89 GB | 6.11 GB | **0.076** | **87.1%** | 12.28 | **54.9%** |

The "types" layout recovers 4.4 of the 5.1 lost exam points (56 answers fixed, 38 broken vs the 4-bit baseline, p = 0.08) and matches the original on 3 of 4 subjects; protecting a few whole layers improved fidelity but not the exam. A finer ruler (groups of 128) reaches similar quality at a smaller size, so the two fixes could be combined (not yet tried). Details in `progress5.pdf`.

### Groups of 128 combined with mixed precision (real packed models)

The sensitivity test was re-run on the packed groups-of-128 model (same layers matter most, but about 4x less: layer 1 changes the output by KL 0.015 instead of 0.059). `plan_mixed.py 0.5 _g128` builds layouts with a +0.5 GB budget.

| 4-bit model (all real packed) | Total size | KL from original | Same top word | Perplexity | CS exam, 412 q |
|---|---|---|---|---|---|
| Original | 15 GB | - | 100% | 11.44 | 55.6% |
| Per row | 5.22 GB | 0.281 | 74.3% | 14.70 | 50.5% |
| Per row + v, k, o, down at 8-bit | 6.11 GB | 0.076 | 87.1% | 12.28 | 54.9% |
| Groups of 128 | 5.61 GB | 0.085 | 86.4% | 12.29 | 54.1% |
| **Groups of 128 + layers 0, 1, 31 at 8-bit** | 5.92 GB | **0.048** | **89.8%** | 12.09 | 54.1% |
| Groups of 128 + v, k, o (and half of q) at 8-bit | 6.11 GB | 0.064 | 88.6% | 12.07 | 54.6% |

The combination gives the best fidelity of any 4-bit model (KL 0.048), at a smaller size than the per-row mixed-precision model. The exam sits at its noise floor (about 2.5 points) for all of these, so it can no longer separate them. **Size note:** the packed groups-of-128 model is 5.61 GB, not the roughly 5.3 GB earlier estimated, because the packed format stores 32-bit scales and zero-points; a leaner format would be about 5.3 GB. `evaluation/plot_results.py` regenerates all figures (`results/figures/`) and `results/summary.md`; `evaluation/make_details_pdf.py` builds `details.pdf`.

Notes: the earlier partial-quantization numbers (only 70/225 layers) are kept in `results/archive/`. Fake-quantized models run with disk offload, so their tokens/sec are not comparable and are not a result. Run `python quantization/quantize_checkpoint.py --bits N` then `python quantization/evaluate_checkpoint.py ...` (see its docstring).

## Hardware

Developed and run on an RTX 3060 Ti (8GB VRAM), Ryzen 5 5600, 16GB RAM.