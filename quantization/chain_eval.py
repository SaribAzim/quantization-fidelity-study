"""
Evaluate a saved model folder layer by layer (fast, no offloading) on the
same WikiText-2 test text as evaluate.py, at a chosen window length.

Used to answer: does the saved GPTQ model score the same as in the main
harness (window 512), and does it behave differently on longer windows (2048,
the length it was calibrated on)?

Run: python quantization/chain_eval.py --model quantized_models/gptq_4bit_g128 --seqlen 512 2048
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantization.quantize_checkpoint import find_cached_snapshot
from quantization.run_gptq import DEVICE, CheckpointReader, build_layer


@torch.no_grad()
def chain_perplexity(model_dir, tokenizer, config, seqlen, num_samples=50):
    reader = CheckpointReader(model_dir)
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"] if t.strip() != ""][:num_samples]
    ids = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids[0]
    windows = [ids[s:s + seqlen] for s in range(0, len(ids), seqlen) if len(ids[s:s + seqlen]) >= 2]

    embed = reader.get("model.embed_tokens.weight").to(DEVICE)
    hidden = [F.embedding(w.to(DEVICE), embed).unsqueeze(0) for w in windows]
    del embed
    rotary = LlamaRotaryEmbedding(config=config).to(DEVICE)
    pes = [rotary(h, torch.arange(h.size(1), device=DEVICE).unsqueeze(0)) for h in hidden]
    pos = [torch.arange(h.size(1), device=DEVICE).unsqueeze(0) for h in hidden]

    for layer_idx in range(config.num_hidden_layers):
        layer = build_layer(config, reader, layer_idx)
        for i in range(len(hidden)):
            out = layer(hidden[i], attention_mask=None, position_ids=pos[i], position_embeddings=pes[i])
            hidden[i] = out[0] if isinstance(out, tuple) else out
        del layer
        torch.cuda.empty_cache()

    norm_w = reader.get("model.norm.weight").to(DEVICE).float()
    lm_head = reader.get("lm_head.weight").to(DEVICE)
    total_loss, total_tokens = 0.0, 0
    for h, w in zip(hidden, windows):
        h = h.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + config.rms_norm_eps) * norm_w
        logits = (h.to(torch.bfloat16) @ lm_head.T).float()
        loss = F.cross_entropy(logits[0, :-1], w[1:].to(DEVICE))
        total_loss += loss.item() * len(w)  # same weighting as evaluate.py
        total_tokens += len(w)
    return torch.tensor(total_loss / total_tokens).exp().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seqlen", type=int, nargs="+", default=[512])
    args = parser.parse_args()

    src_dir = find_cached_snapshot()
    config = AutoConfig.from_pretrained(src_dir)
    config._attn_implementation = "sdpa"
    tokenizer = AutoTokenizer.from_pretrained(src_dir)
    model_dir = src_dir if args.model == "base" else args.model
    for seqlen in args.seqlen:
        ppl = chain_perplexity(model_dir, tokenizer, config, seqlen)
        print(f"RESULT {args.model} window {seqlen}: perplexity {ppl:.3f}", flush=True)


if __name__ == "__main__":
    main()
