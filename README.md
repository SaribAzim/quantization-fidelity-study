Quantization Fidelity Study

Research question: How much does an 8B parameter LLM degrade when quantized, and can a RAG architecture recover the fidelity lost in a specific domain?

Model

Llama 3.1 8B Instruct (Meta, via Hugging Face)

Project structure
quantization/   - Our own PTQ and GPTQ-style quantization implementations
evaluation/     - Evaluation harness (perplexity, task accuracy, latency, memory)
rag/            - RAG pipeline built on the quantized model
deployment/     - FastAPI service + Docker setup
results/        - Logged metrics and comparison tables from every phase
notebooks/      - Exploratory work, not the source of truth for final code
Evaluation metrics

Every phase (base model, naive-quantized, GPTQ-quantized, RAG-augmented) is measured on the same four metrics, so comparisons stay apples-to-apples:

Perplexity — language modeling quality on held-out text
Task accuracy — MMLU subset, reasoning/knowledge performance
Latency — tokens per second
Memory footprint — peak VRAM usage and model size on disk
Status

Project in progress. This README will be updated with results as each phase completes.

Hardware

Developed and run on an RTX 3060 Ti (8GB VRAM), Ryzen 5 5600, 16GB RAM.