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


# How many of the reference model's most likely tokens we keep per position
# when measuring fidelity (see evaluate_perplexity_and_fidelity).
FIDELITY_TOP_K = 256


def evaluate_perplexity_and_fidelity(
    model, tokenizer, reference_path, save_reference=False, num_samples=50, stride=512
):
    """
    One pass over the SAME WikiText-2 slice and windows as evaluate_perplexity,
    but it also measures how closely this model's predictions match the
    original (base) model's, which is the actual question of this project.

    WHY THIS EXISTS: perplexity is one number about "how well does the model
    predict real text". Two models can have the same perplexity while
    disagreeing on many individual predictions. Fidelity asks directly:
      - top-1 agreement: how often does this model's single most likely next
        token equal the base model's? (1.0 = identical choices)
      - KL divergence: how far is this model's whole probability
        distribution from the base model's, averaged over positions?
        (0.0 = identical distributions, bigger = more different)

    HOW: we run the base model once with save_reference=True. For every
    position it stores the FIDELITY_TOP_K most likely tokens and their
    log-probabilities (storing the whole 128k-token distribution for every
    position would take gigabytes). Later models are compared against that
    file. The KL is computed over those top-K tokens plus one "everything
    else" bucket, which covers essentially all the probability mass.

    The perplexity computed here is identical to evaluate_perplexity: same
    text, same windows, same loss, same averaging.

    Returns a dict with perplexity, kl_divergence, top1_agreement and
    num_tokens. kl/top1 are None when this call is creating the reference.
    """
    import os

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"] if t.strip() != ""][:num_samples]
    full_text = "\n\n".join(texts)
    input_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)

    reference = None
    if not save_reference:
        reference = torch.load(reference_path)
    saved_idx, saved_logp = [], []
    ref_offset = 0

    total_log_likelihood = 0.0
    total_tokens = 0
    total_kl = 0.0
    total_agree = 0
    total_positions = 0

    seq_len = input_ids.size(1)
    print(f"Running perplexity + fidelity over {seq_len} tokens...")
    for start in range(0, seq_len, stride):
        chunk = input_ids[:, start:min(start + stride, seq_len)]
        if chunk.size(1) < 2:
            continue  # a 1-token window has nothing to predict

        with torch.no_grad():
            outputs = model(chunk, labels=chunk)

        total_log_likelihood += outputs.loss.item() * chunk.size(1)
        total_tokens += chunk.size(1)

        # Position i's logits predict token i+1, so the last position has no
        # target inside this window and is dropped.
        logits = outputs.logits[0, :-1].float()
        log_norm = torch.logsumexp(logits, dim=-1, keepdim=True)
        num_positions = logits.size(0)

        if save_reference:
            top_logits, top_idx = logits.topk(FIDELITY_TOP_K, dim=-1)
            saved_idx.append(top_idx.cpu().int())
            saved_logp.append((top_logits - log_norm).cpu())
        else:
            ref_idx = reference["top_idx"][ref_offset:ref_offset + num_positions].to(logits.device).long()
            ref_logp = reference["top_logp"][ref_offset:ref_offset + num_positions].to(logits.device)
            ref_offset += num_positions

            cand_logp = logits.gather(-1, ref_idx) - log_norm  # same tokens, this model's log-probs

            # Probability left over outside the top-K tokens, per model.
            ref_tail = (1.0 - ref_logp.exp().sum(-1)).clamp(min=1e-9)
            cand_tail = (1.0 - cand_logp.exp().sum(-1)).clamp(min=1e-9)
            kl = (ref_logp.exp() * (ref_logp - cand_logp)).sum(-1) + ref_tail * (ref_tail.log() - cand_tail.log())

            total_kl += kl.sum().item()
            # Column 0 is the reference's most likely token (topk is sorted).
            total_agree += (logits.argmax(-1) == ref_idx[:, 0]).sum().item()
            total_positions += num_positions

    perplexity = torch.exp(torch.tensor(total_log_likelihood / total_tokens)).item()

    if save_reference:
        os.makedirs(os.path.dirname(reference_path) or ".", exist_ok=True)
        torch.save({"top_idx": torch.cat(saved_idx), "top_logp": torch.cat(saved_logp)}, reference_path)
        print(f"Saved fidelity reference to {reference_path}")
        return {"perplexity": perplexity, "kl_divergence": None, "top1_agreement": None, "num_tokens": total_tokens}

    assert ref_offset == reference["top_idx"].size(0), "reference was made from different text/windows"
    return {
        "perplexity": perplexity,
        "kl_divergence": total_kl / total_positions,
        "top1_agreement": total_agree / total_positions,
        "num_tokens": total_tokens,
    }


def is_fully_on_gpu(model):
    """
    True if every part of the model lives on the GPU. When accelerate has to
    spill layers to CPU RAM or disk, each generated token drags those weights
    back through the GPU, so tokens/sec then mostly measures disk and RAM
    speed rather than the model. Such latency numbers are not comparable.
    """
    device_map = getattr(model, "hf_device_map", None)
    if device_map is None:
        return next(model.parameters()).is_cuda
    return all(isinstance(d, int) or str(d).startswith("cuda") for d in device_map.values())


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
    # Count real bytes: numel * bytes-per-element for every parameter AND
    # buffer. Buffers matter because packed GPTQ models store their 4-bit
    # weights in buffers (qweight, scales, zeros), not in parameters, and
    # the old "params * 2 bytes" rule would have both ignored them and
    # reported a quantized model as if it were BF16.
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_gb = total_bytes / (1024 ** 3)

    peak_vram_gb = None
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return {"model_size_gb": model_size_gb, "peak_vram_gb": peak_vram_gb}


def estimate_packed_size_gb(model, num_bits, group_size=-1):
    """
    How big this model WOULD be on disk if the quantized Linear weights were
    stored packed at num_bits (fake-quantized models are stored in BF16, so
    their real file size says nothing about the saving).

    Per quantized weight: numel * num_bits / 8 bytes, plus the scale (2 bytes,
    BF16) and zero-point (num_bits/8 bytes) for each row (group_size=-1) or
    each group of `group_size` input columns. Everything else (embeddings,
    lm_head, norms) stays BF16.
    """
    from quantization.quantize_checkpoint import is_quantizable

    total_bytes = 0.0
    for name, p in model.named_parameters():
        if is_quantizable(name):
            rows, cols = p.shape
            groups_per_row = 1 if group_size == -1 else cols // group_size
            total_bytes += p.numel() * num_bits / 8
            total_bytes += rows * groups_per_row * (2 + num_bits / 8)
        else:
            total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024 ** 3)


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