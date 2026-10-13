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

Notes: the earlier partial-quantization numbers (only 70/225 layers) are kept in `results/archive/`. Fake-quantized models run with disk offload, so their tokens/sec are not comparable and are not a result. Run `python quantization/quantize_checkpoint.py --bits N` then `python quantization/evaluate_checkpoint.py ...` (see its docstring).

## Hardware

Developed and run on an RTX 3060 Ti (8GB VRAM), Ryzen 5 5600, 16GB RAM.