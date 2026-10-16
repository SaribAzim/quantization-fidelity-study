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

Notes: the earlier partial-quantization numbers (only 70/225 layers) are kept in `results/archive/`. Fake-quantized models run with disk offload, so their tokens/sec are not comparable and are not a result. Run `python quantization/quantize_checkpoint.py --bits N` then `python quantization/evaluate_checkpoint.py ...` (see its docstring).

## Hardware

Developed and run on an RTX 3060 Ti (8GB VRAM), Ryzen 5 5600, 16GB RAM.