"""
QLoRA on our own packed 4-bit RTN model.

The 4-bit weights stay frozen (and packed, about 5.2 GB on the GPU). We train
only small LoRA adapters (about 42 million numbers for rank 16) that learn to
undo part of the rounding damage. The objective is the ordinary language-
model one: predict the next word of real text.

Choices that matter (and why):
  * Training text is WikiText-2 TRAIN, cut into 512-token windows with no
    special start token, exactly the shape of the test windows. (GPTQ taught
    us that a method tuned on one text shape can fail on another.)
  * The exam questions and the test text are never used for training.
  * Gradient checkpointing: instead of remembering every layer's internals for
    the backward pass, recompute them layer by layer. Slower, but it is what
    lets training fit in 8 GB.
  * The 1 GB word-embedding table lives in CPU RAM (it is frozen, only looked up).
  * Fixed settings chosen BEFORE seeing exam results; we do not tune on the exam.

Run from the project root:
    python quantization/train_qlora.py --smoke               (3 quick steps: memory, speed, sanity)
    python quantization/train_qlora.py --max-tokens 1500000
"""

import argparse
import json
import math
import os
import sys
import time

import torch
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantization.packed import add_lora_to_model, load_packed_model, lora_state


def window_tokens(tokenizer, split, seqlen, max_windows=None):
    """WikiText-2 `split` as consecutive non-overlapping windows of seqlen tokens: (n, seqlen)."""
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    ids = tokenizer("\n\n".join(t for t in data["text"] if t.strip()), return_tensors="pt", add_special_tokens=False).input_ids[0]
    n = len(ids) // seqlen
    windows = ids[:n * seqlen].view(n, seqlen)
    return windows[:max_windows] if max_windows else windows


@torch.no_grad()
def validation_loss(model, windows, batch_size=2):
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size].cuda()
        total += model(input_ids=batch, labels=batch).loss.item() * len(batch)
        count += len(batch)
    model.train()
    return total / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="quantized_models/packed_naive_4bit")
    parser.add_argument("--out", default="quantized_models/qlora/adapters_r16.pt")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--accum", type=int, default=16, help="micro-batches per optimizer step")
    parser.add_argument("--max-tokens", type=int, default=1_500_000)
    parser.add_argument("--max-minutes", type=float, default=None, help="stop (and save) after this many minutes")
    parser.add_argument("--val-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="3 tiny steps to check memory and speed")
    parser.add_argument("--log", default="results/qlora/train_log.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model, tokenizer = load_packed_model(args.model, embeddings_on_cpu=True)
    params = add_lora_to_model(model, args.rank, args.alpha)
    print(f"Trainable adapter parameters: {sum(p.numel() for p in params) / 1e6:.1f} M "
          f"(rank {args.rank}); frozen packed model on GPU: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB", flush=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    windows = window_tokens(tokenizer, "train", args.seqlen)
    val_windows = window_tokens(tokenizer, "validation", args.seqlen, max_windows=40)
    order = torch.randperm(len(windows), generator=torch.Generator().manual_seed(args.seed))
    tokens_per_step = args.accum * args.micro_batch * args.seqlen
    total_steps = 3 if args.smoke else max(1, args.max_tokens // tokens_per_step)
    accum = 2 if args.smoke else args.accum
    warmup = max(1, int(0.05 * total_steps))
    print(f"{len(windows)} training windows available; {total_steps} steps x {tokens_per_step} tokens = "
          f"{total_steps * tokens_per_step / 1e6:.2f} M tokens (one pass uses {total_steps * args.accum * args.micro_batch / len(windows):.0%} of the data)", flush=True)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    lr_at = lambda s: args.lr * (s + 1) / warmup if s < warmup else args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))))

    log, start, cursor = [], time.time(), 0
    start_val = validation_loss(model, val_windows)
    print(f"validation loss before any training: {start_val:.4f} (perplexity {math.exp(start_val):.2f})", flush=True)
    log.append({"step": 0, "val_loss": start_val})
    torch.cuda.reset_peak_memory_stats()

    for step in range(total_steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        t0, step_loss = time.time(), 0.0
        for _ in range(accum):
            idx = order[cursor % len(order):cursor % len(order) + args.micro_batch]
            cursor += args.micro_batch
            batch = windows[idx].cuda()
            loss = model(input_ids=batch, labels=batch).loss / accum
            loss.backward()
            step_loss += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        entry = {"step": step + 1, "train_loss": step_loss, "lr": lr_at(step), "grad_norm": grad_norm,
                 "tokens_per_second": accum * args.micro_batch * args.seqlen / (time.time() - t0),
                 "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024 ** 3}
        if (step + 1) % args.val_every == 0 or step + 1 == total_steps:
            entry["val_loss"] = validation_loss(model, val_windows)
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save(lora_state(model), args.out)
        log.append(entry)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in entry.items()}), flush=True)

        if args.max_minutes and (time.time() - start) / 60 > args.max_minutes:
            print(f"Time budget of {args.max_minutes} min reached after {step + 1} steps; saving.", flush=True)
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save(lora_state(model), args.out)
            break

    if not args.smoke:
        os.makedirs(os.path.dirname(args.log), exist_ok=True)
        json.dump({"config": vars(args), "log": log, "seconds": time.time() - start}, open(args.log, "w"), indent=1)
    final_val = validation_loss(model, val_windows)
    print(f"\nDone in {(time.time() - start) / 60:.1f} min. Validation loss {start_val:.4f} -> {final_val:.4f} "
          f"(perplexity {math.exp(start_val):.2f} -> {math.exp(final_val):.2f}); peak VRAM {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
