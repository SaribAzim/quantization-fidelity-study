"""
HyDE = Hypothetical Document Embeddings.

THE PROBLEM IT SOLVES: a short question ("What does x << 3 do?") looks very
different from the encyclopedia text that answers it, so searching with the
question can miss the right passage.

THE TRICK: first ask the language model to WRITE a short made-up answer
passage. That passage may contain mistakes, but it LOOKS like the real
documents (same style, same vocabulary), so searching with it lands closer to
the right passages. The made-up text is thrown away afterwards; only its
embedding is used to search.

In this project the model that writes the made-up passage is the QUANTIZED
model, so a damaged model could write worse hypothetical passages. That is
part of what we want to measure.
"""

import json
import os

import torch

from rag.mcq import LETTERS

HYDE_INSTRUCTION = (
    "Write a short, factual passage (about three sentences) in the style of a computer science "
    "textbook or encyclopedia article that would help someone answer the question below. "
    "Explain the relevant concepts. Do not list the options and do not say which option is correct."
)


def hyde_prompt_messages(q):
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, q["choices"]))
    return [
        {"role": "system", "content": "You are an expert computer science teacher."},
        {"role": "user", "content": f"{HYDE_INSTRUCTION}\n\nQuestion: {q['question']}\nOptions:\n{options}"},
    ]


@torch.no_grad()
def generate_hypothetical_passages(model, tokenizer, questions, cache_path, batch_size=32, max_new_tokens=80):
    """
    Returns {"subject/doc_id": passage}. Results are cached on disk after every
    batch, so an interrupted run resumes where it stopped.
    """
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    todo = [q for q in questions if f"{q['subject']}/{q['doc_id']}" not in cache]
    print(f"HyDE: {len(cache)} passages cached, {len(todo)} to generate", flush=True)

    tokenizer.padding_side = "left"  # generation continues from the right edge, so pad on the left
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    start = 0
    while start < len(todo):
        batch = todo[start:start + batch_size]
        prompts = [tokenizer.apply_chat_template(hyde_prompt_messages(q), tokenize=False, add_generation_prompt=True)
                   for q in batch]
        try:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            print(f"  out of GPU memory, retrying with batch size {batch_size}", flush=True)
            continue
        texts = tokenizer.batch_decode(output[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
        for q, text in zip(batch, texts):
            cache[f"{q['subject']}/{q['doc_id']}"] = text.strip()
        json.dump(cache, open(cache_path, "w"), indent=1)
        start += len(batch)
        print(f"  generated {start}/{len(todo)}", flush=True)
    return cache
