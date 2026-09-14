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


if __name__ == "__main__":
    import json
    import os
    from load_model import load_base_model

    model, tokenizer = load_base_model()
    ppl = evaluate_perplexity(model, tokenizer)
    print(f"\nBase model perplexity on WikiText-2 (first 50 samples): {ppl:.4f}")

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
        "mmlu_task": "high_school_computer_science",
        "mmlu_accuracy": 0.68,
        "mmlu_stderr": 0.0666,
        "mmlu_num_samples": 50,
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {results_path}")