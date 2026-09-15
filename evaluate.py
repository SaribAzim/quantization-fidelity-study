"""
Phase 1 - Evaluation harness, piece 1: Perplexity.

This file is designed to be reused in Phase 2 and 3 too, evaluate_perplexity
takes any already-loaded model and tokenizer, so it doesn't care whether
that model is the base model, a naively-quantized one, or a GPTQ one.
That's why this is separate from load_model.py.

WHAT IS PERPLEXITY, actually:
A language model doesn't just generate text, at every position it produces
a probability distribution over "what token comes next." Perplexity measures
how well the model predicted the ACTUAL next tokens in a piece of real text
it has never seen during training.

Concretely: for each token in our test text, we ask the model "how much
probability did you assign to this being the correct next token?" We
average that across the whole text (in log-space, for numerical reasons),
then convert back with exp(). The result is perplexity.

Intuition for reading the number:
- Perplexity of 1 would mean the model was 100% certain about every single
  next token, and always right. This never happens in practice.
- Lower perplexity = the model was, on average, less "surprised" by the
  real text = better language modeling.
- Higher perplexity = the model found the real text less predictable.

Why we care for THIS project specifically: if quantization hurts a model,
one of the first places it shows up is here, the model gets less confident
and less accurate about predicting real text, so perplexity goes up.
"""

import torch
from datasets import load_dataset


def evaluate_perplexity(model, tokenizer, num_samples=50, stride=512):
    """
    Computes perplexity on a held-out text dataset (WikiText-2).

    Args:
        model: an already-loaded HuggingFace causal LM (base or quantized)
        tokenizer: the matching tokenizer
        num_samples: how many text samples from the dataset to evaluate on.
            We don't use the whole dataset, that would take very long, we
            just need a consistent, repeatable slice we can compare against
            the same slice later for the quantized model.
        stride: how many tokens we feed the model at once. This exists
            because a full document can be much longer than the model's
            context window. We chop it into overlapping windows of this size.

    Returns:
        A single float: the perplexity score.
    """
    print("Loading WikiText-2 test split...")
    # Note: the dataset is loaded via its namespaced repo ID (Salesforce/wikitext),
    # not the older bare "wikitext" name. Newer versions of huggingface_hub
    # require all dataset repo IDs to be in "namespace/name" format, and reject
    # bare names outright. Same dataset, same content, just the current valid ID.
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

    # Join a fixed number of non-empty text samples into one long string.
    # Using a FIXED number of samples (not the whole dataset) is what makes
    # this comparable across models later, base and quantized both see
    # exactly this same text.
    texts = [t for t in dataset["text"] if t.strip() != ""][:num_samples]
    full_text = "\n\n".join(texts)

    # Tokenize the whole thing into one long sequence of token IDs.
    encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)

    total_log_likelihood = 0.0
    total_tokens = 0

    print(f"Running perplexity evaluation over {input_ids.size(1)} tokens...")

    # We slide a window across the tokenized text, `stride` tokens at a time.
    # Why not just feed everything at once: the model has a maximum context
    # length, and very long sequences also cost a lot of memory. Sliding
    # window evaluation is the standard way to handle text longer than that.
    seq_len = input_ids.size(1)
    for start in range(0, seq_len, stride):
        end = min(start + stride, seq_len)
        chunk = input_ids[:, start:end]

        with torch.no_grad():
            # Passing labels=chunk makes the model compute its own
            # cross-entropy loss internally, which IS the average
            # negative log-likelihood per token, exactly what we need.
            outputs = model(chunk, labels=chunk)
            loss = outputs.loss  # average negative log-likelihood for this chunk

        num_tokens_in_chunk = chunk.size(1)
        total_log_likelihood += loss.item() * num_tokens_in_chunk
        total_tokens += num_tokens_in_chunk

    # Average negative log-likelihood across every token we evaluated,
    # then exponentiate to convert from log-space back to perplexity.
    average_negative_log_likelihood = total_log_likelihood / total_tokens
    perplexity = torch.exp(torch.tensor(average_negative_log_likelihood)).item()

    return perplexity


def evaluate_latency(model, tokenizer, prompt="Explain the concept of gravity in simple terms.", max_new_tokens=50):
    """
    Measures generation speed in tokens per second.

    Why a warm-up generation happens first, untimed: the first call to
    model.generate() on a GPU includes one-time setup costs (CUDA kernels
    compiling, memory allocator warming up) that have nothing to do with
    the model's actual steady-state speed. Timing that first call would
    make the model look artificially slower than it really is. So we
    generate once without timing it, then time a second, real generation.

    Why torch.cuda.synchronize() matters here: GPU operations in PyTorch
    are launched asynchronously, meaning Python's timer can move on before
    the GPU actually finishes the work. Without synchronize(), we'd measure
    "how long it took to launch the work" instead of "how long the work
    actually took." Calling synchronize() blocks until the GPU is truly done,
    making our timing accurate.

    Returns:
        A float: tokens generated per second.
    """
    import time

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("Running warm-up generation (untimed)...")
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=5, do_sample=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print("Running timed generation...")
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()

    # output_ids includes the original prompt tokens plus the newly
    # generated ones, so we subtract the prompt length to count only
    # what was actually generated during our timed window.
    num_new_tokens = output_ids.shape[1] - inputs["input_ids"].shape[1]
    elapsed_seconds = end_time - start_time

    tokens_per_second = num_new_tokens / elapsed_seconds
    return tokens_per_second


def evaluate_memory_footprint(model):
    """
    Measures two things: the model's size on disk (calculated from its
    actual parameter count, not by checking file sizes, since that stays
    accurate regardless of how accelerate splits the model across GPU/CPU/
    disk), and the peak GPU memory used during the most recent inference.

    Why we reset peak memory stats before calling this, elsewhere in the
    script: torch.cuda.max_memory_allocated() reports the highest memory
    usage since the last reset, not "right now." If we didn't reset it
    before running generation, this number would be contaminated by
    memory used during model LOADING, which behaves differently (and
    less representatively of real usage) than actual inference.

    Returns:
        A dict with "model_size_gb" and "peak_vram_gb".
    """
    total_params = sum(p.numel() for p in model.parameters())

    # BF16 uses 2 bytes per parameter. This calculation stays correct even
    # for the quantized models in later phases, we'll just pass a different
    # bytes-per-parameter value at that point (e.g. 0.5 for 4-bit).
    bytes_per_param = 2
    model_size_gb = (total_params * bytes_per_param) / (1024 ** 3)

    peak_vram_gb = None
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return {"model_size_gb": model_size_gb, "peak_vram_gb": peak_vram_gb}


if __name__ == "__main__":
    import json
    import os
    from load_model import load_base_model

    model, tokenizer = load_base_model()
    ppl = evaluate_perplexity(model, tokenizer)
    print(f"\nBase model perplexity on WikiText-2 (first 50 samples): {ppl:.4f}")

    # Reset peak memory tracking HERE, right before the measurements we
    # actually care about (latency, memory footprint), so loading-time
    # memory usage doesn't contaminate the peak we report.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    tokens_per_second = evaluate_latency(model, tokenizer)
    print(f"Base model latency: {tokens_per_second:.2f} tokens/sec")

    memory_stats = evaluate_memory_footprint(model)
    print(f"Base model size on disk: {memory_stats['model_size_gb']:.2f} GB")
    print(f"Peak VRAM during generation: {memory_stats['peak_vram_gb']:.2f} GB")

    # MMLU task accuracy is evaluated separately, via lm-evaluation-harness
    # (see README for the exact command), not through this script. An
    # earlier hand-rolled version lived here, but comparing raw logits at
    # a single hand-picked token position proved too fragile for an
    # instruction-tuned model, small tokenizer quirks threw the score off
    # by a large margin. lm-eval handles chat formatting and answer
    # parsing correctly, so we use it instead and record its result
    # directly into results/baseline.json.

    # Save the result so it persists beyond this terminal session, and so
    # Phase 2/3 can load this same file and append their own numbers for
    # direct comparison, rather than us copy-pasting numbers by hand.
    os.makedirs("results", exist_ok=True)
    results_path = "results/baseline.json"

    results = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "precision": "bf16",
        "perplexity": ppl,
        "tokens_per_second": tokens_per_second,
        "model_size_gb": memory_stats["model_size_gb"],
        "peak_vram_gb": memory_stats["peak_vram_gb"],
        "mmlu_task": "high_school_computer_science",
        "mmlu_accuracy": 0.68,
        "mmlu_stderr": 0.0666,
        "mmlu_num_samples": 50,
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {results_path}")