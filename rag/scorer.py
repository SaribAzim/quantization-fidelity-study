"""
Answers the exam questions, optionally with retrieved passages in the prompt.

lm-evaluation-harness cannot add retrieved text to its prompts, so we score
the questions ourselves, copying exactly what it does:
  - the prompt is the model's chat template with the subject line as system
    message and the question plus options as the user message;
  - the model's answer is whichever of the letters A/B/C/D it considers most
    likely as the first word of its reply.
With retrieval switched off this must reproduce lm-eval's per-question
results; run_rag_eval.py checks that before any RAG number is trusted.
"""

import torch

from rag.mcq import LETTERS, question_block, subject_description


def build_prompt(tokenizer, q, passages=None):
    """The chat-formatted prompt text for one question; `passages` is a list of passage strings or None."""
    user = question_block(q)
    if passages:
        refs = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
        user = f"Reference passages (use them if they help):\n\n{refs}\n\n---\n\n{user}"
    messages = [{"role": "system", "content": subject_description(q["subject"])},
                {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def letter_token_ids(tokenizer, prompt):
    """Token id of each answer letter as it appears right after the prompt (checked, not assumed)."""
    ids = []
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    for letter in LETTERS:
        whole = tokenizer(prompt + letter, add_special_tokens=False)["input_ids"]
        continuation = whole[len(prompt_ids):]
        assert len(continuation) == 1, f"letter {letter} is not a single token: {continuation}"
        ids.append(continuation[0])
    return ids


@torch.no_grad()
def score_prompts(model, tokenizer, prompts, batch_size=16):
    """
    Returns, for every prompt, the model's log-probabilities of A, B, C, D.

    Prompts are sorted by length so a batch holds similar lengths (less
    padding). Padding goes on the left; the position encoding the model uses
    only looks at DISTANCES between words, so the padding does not change the
    answer, and the pad positions are masked out.
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    letter_ids = letter_token_ids(tokenizer, prompts[0])

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    results = [None] * len(prompts)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(model.device)
        # logits_to_keep=1: only compute the next-word scores at the LAST position.
        logits = model(**batch, logits_to_keep=1).logits[:, -1].float()
        letter_logprobs = torch.log_softmax(logits, dim=-1)[:, letter_ids].cpu()
        for row, i in enumerate(idx):
            results[i] = letter_logprobs[row].tolist()
        print(f"  scored {min(start + batch_size, len(order))}/{len(order)}", flush=True)
    return results
