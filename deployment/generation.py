"""
Token-by-token greedy generation, shared by the gallery recorder and the live
server, so recorded and live answers are produced in exactly the same way.

Greedy means the model always takes its single most likely next token, so the
same prompt always gives the same answer.
"""

import time

import torch

STOP_IDS = {128001, 128008, 128009}  # Llama 3.1 end of text, end of message and end of turn tokens


@torch.no_grad()
def stream_answer(model, tokenizer, prompt, max_new_tokens, should_stop=lambda: False):
    """
    Yields (token_id, new_text, seconds_since_start) for every generated token.
    `new_text` is the text this token adds (whole characters, even when one
    character spans several tokens). `should_stop` lets a caller cancel early.
    """
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    current = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    out_ids, decoded, past = [], "", None
    start = time.time()
    for _ in range(max_new_tokens):
        if should_stop():
            return
        out = model(input_ids=current, past_key_values=past, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        token = int(out.logits[0, -1].argmax())
        if token in STOP_IDS:
            return
        out_ids.append(token)
        full = tokenizer.decode(out_ids, skip_special_tokens=True)
        new_text, decoded = full[len(decoded):], full
        yield token, new_text, round(time.time() - start, 3)
        current = torch.tensor([[token]], device=model.device)
