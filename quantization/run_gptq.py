"""
Phase 3 - Run GPTQ on Llama 3.1 8B Instruct, one decoder layer at a time.

WHY LAYER BY LAYER: the whole model (15GB) does not fit on the GPU, and we
cannot load it live (that is the meta-tensor trap from Phase 2). So we never
build the model. We keep only:
  - the calibration activations (the numbers flowing between layers), on CPU
  - ONE decoder layer on the GPU at a time, loaded straight from the
    safetensors shards.

For each of the 32 layers:
  1. load its weights, run the calibration text through it, and record what
     goes into q/k/v, o, gate/up and down (the Hessians, see gptq.py)
  2. GPTQ-quantize its 7 Linear weights
  3. run the calibration text through the QUANTIZED layer; these outputs are
     the next layer's inputs, so each layer learns to correct the damage
     already done by the layers before it
  4. save the layer to its own small .safetensors file

Output is "fake quantized" BF16 like the naive models, so the SAME evaluation
code measures both. The output folder loads with plain from_pretrained.

Calibration text is WikiText-2 *train*, never the *test* text used for
perplexity. Caveat worth knowing: it is the same kind of text as that
perplexity test, so perplexity may flatter GPTQ slightly; the exam (MMLU)
and KL numbers are the fairer checks.

Run from the project root:
    python quantization/run_gptq.py --bits 4 --group-size 128
    python quantization/run_gptq.py --dry-fp --nsamples 8     (checks the layer-by-layer forward pass, no quantization)
"""

import argparse
import json
import os
import shutil
import sys
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantization.gptq import HessianAccumulator, gptq_quantize
from quantization.quantize_checkpoint import MODEL_NAME, SIDECAR_FILES, find_cached_snapshot

DEVICE = "cuda"
LAYER_LINEARS = {  # which Hessian each Linear uses (q,k,v share one input; gate,up share one)
    "self_attn.q_proj": "qkv", "self_attn.k_proj": "qkv", "self_attn.v_proj": "qkv",
    "self_attn.o_proj": "o",
    "mlp.gate_proj": "gate_up", "mlp.up_proj": "gate_up",
    "mlp.down_proj": "down",
}


class CheckpointReader:
    """Reads single tensors from the sharded checkpoint using its index file."""

    def __init__(self, src_dir):
        self.src_dir = src_dir
        self.weight_map = json.load(open(os.path.join(src_dir, "model.safetensors.index.json")))["weight_map"]

    def get(self, name):
        with safe_open(os.path.join(self.src_dir, self.weight_map[name]), framework="pt", device="cpu") as f:
            return f.get_tensor(name)


def load_calibration_ids(tokenizer, nsamples, seqlen, seed):
    """nsamples random windows of seqlen tokens from WikiText-2 train, fixed by seed."""
    train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ids = tokenizer("\n\n".join(train["text"]), return_tensors="pt").input_ids[0]
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(ids) - seqlen - 1, (nsamples,), generator=generator)
    return torch.stack([ids[s:s + seqlen] for s in starts])


def build_layer(config, reader, layer_idx):
    """Builds decoder layer `layer_idx` on the GPU in BF16 with its real weights."""
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(DEVICE):
            layer = LlamaDecoderLayer(config, layer_idx)
    finally:
        torch.set_default_dtype(torch.float32)
    prefix = f"model.layers.{layer_idx}."
    state = {n[len(prefix):]: reader.get(n).to(DEVICE) for n in reader.weight_map if n.startswith(prefix)}
    layer.load_state_dict(state)
    return layer.eval()


def run_layer(layer, inputs, outputs, batch_size, position_ids, position_embeddings, keep_first=False):
    """Pushes the calibration activations through one layer; fills `outputs` (CPU)."""
    first = None
    for b in range(0, inputs.size(0), batch_size):
        x = inputs[b:b + batch_size].to(DEVICE)
        # attention_mask=None: scaled-dot-product attention then applies the causal mask itself.
        out = layer(x, attention_mask=None, position_ids=position_ids, position_embeddings=position_embeddings)
        out = out[0] if isinstance(out, tuple) else out
        outputs[b:b + batch_size] = out.cpu()
        if keep_first and first is None:
            first = out.float()
    return first


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128, help="-1 = per-channel")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--dry-fp", action="store_true", help="no quantization: only check the layer-by-layer forward pass")
    args = parser.parse_args()

    suffix = "" if args.group_size == -1 else f"_g{args.group_size}"
    out_dir = args.out or f"quantized_models/gptq_{args.bits}bit{suffix}"
    src_dir = find_cached_snapshot()
    reader = CheckpointReader(src_dir)
    config = AutoConfig.from_pretrained(src_dir)
    config._attn_implementation = "sdpa"
    tokenizer = AutoTokenizer.from_pretrained(src_dir)

    print(f"Calibration: {args.nsamples} x {args.seqlen} tokens, seed {args.seed}", flush=True)
    ids = load_calibration_ids(tokenizer, args.nsamples, args.seqlen, args.seed)

    # Layer-0 inputs = word embeddings of the calibration text.
    embed = reader.get("model.embed_tokens.weight").to(DEVICE)
    inputs = torch.zeros(args.nsamples, args.seqlen, config.hidden_size, dtype=torch.bfloat16)
    for b in range(0, args.nsamples, args.batch_size):
        inputs[b:b + args.batch_size] = F.embedding(ids[b:b + args.batch_size].to(DEVICE), embed).cpu()
    del embed
    outputs = torch.zeros_like(inputs)

    rotary = LlamaRotaryEmbedding(config=config).to(DEVICE)
    position_ids = torch.arange(args.seqlen, device=DEVICE).unsqueeze(0)
    position_embeddings = rotary(inputs[:1].to(DEVICE), position_ids)

    if not args.dry_fp:
        os.makedirs(out_dir, exist_ok=True)
    weight_map, log = {}, []
    start_time = time.time()

    for layer_idx in range(config.num_hidden_layers):
        t0 = time.time()
        layer = build_layer(config, reader, layer_idx)
        record = {"layer": layer_idx}

        if not args.dry_fp:
            # Pass 1: record the Hessians while the FULL-precision layer runs.
            accumulators, hooks = {}, []
            modules = {name: layer.get_submodule(name) for name in LAYER_LINEARS}
            for name, key in LAYER_LINEARS.items():
                if key not in accumulators:
                    accumulators[key] = HessianAccumulator(modules[name].in_features, DEVICE)
                    hooks.append(modules[name].register_forward_hook(
                        lambda mod, inp, out, acc=accumulators[key]: acc.add_batch(inp[0])))
            fp_first = run_layer(layer, inputs, outputs, args.batch_size, position_ids, position_embeddings, keep_first=True)
            for h in hooks:
                h.remove()

            # Quantize all 7 Linears of this layer.
            for name, key in LAYER_LINEARS.items():
                weight = modules[name].weight.data
                q = gptq_quantize(weight.float(), accumulators[key].H, args.bits, group_size=args.group_size)
                modules[name].weight.data = q.to(weight.dtype)
            del accumulators

        # Pass 2: run the (now quantized) layer; its outputs feed the next layer.
        q_first = run_layer(layer, inputs, outputs, args.batch_size, position_ids, position_embeddings, keep_first=True)
        if not args.dry_fp:
            record["rel_output_error"] = ((q_first - fp_first).norm() / fp_first.norm()).item()
            tensors = {f"model.layers.{layer_idx}.{n}": t.detach().cpu().contiguous() for n, t in layer.state_dict().items()}
            file_name = f"model-layer-{layer_idx:02d}.safetensors"
            save_file(tensors, os.path.join(out_dir, file_name), metadata={"format": "pt"})
            weight_map.update({n: file_name for n in tensors})

        inputs, outputs = outputs, inputs
        del layer
        torch.cuda.empty_cache()
        record["seconds"] = round(time.time() - t0, 1)
        log.append(record)
        print(record, flush=True)

    # Sanity check of the layer-by-layer forward: final norm + lm_head on the
    # first 2 calibration samples. A sane perplexity (roughly 5-15 on wikitext
    # train text) proves the hand-built forward pass is right; a broken mask or
    # rotary embedding gives nonsense (near 1 or in the thousands).
    norm_w = reader.get("model.norm.weight").to(DEVICE).float()
    lm_head = reader.get("lm_head.weight").to(DEVICE)
    h = inputs[:2].to(DEVICE).float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + config.rms_norm_eps) * norm_w
    logits = (h.to(torch.bfloat16) @ lm_head.T).float()
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), ids[:2, 1:].reshape(-1).to(DEVICE))
    label = "FULL-PRECISION layer-by-layer" if args.dry_fp else f"GPTQ {args.bits}-bit"
    print(f"\n{label} perplexity on 2 calibration samples: {loss.exp().item():.2f}", flush=True)
    if args.dry_fp:
        return

    extras = {"model.norm.weight": reader.get("model.norm.weight"), "lm_head.weight": reader.get("lm_head.weight"),
              "model.embed_tokens.weight": reader.get("model.embed_tokens.weight")}
    save_file({k: v.contiguous() for k, v in extras.items()}, os.path.join(out_dir, "model-extras.safetensors"), metadata={"format": "pt"})
    weight_map.update({k: "model-extras.safetensors" for k in extras})

    total_size = sum(t.numel() * t.element_size() for t in extras.values())
    json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map},
              open(os.path.join(out_dir, "model.safetensors.index.json"), "w"), indent=2)
    for side_file in SIDECAR_FILES:
        src = os.path.join(src_dir, side_file)
        if os.path.exists(src) and side_file != "model.safetensors.index.json":
            shutil.copyfile(src, os.path.join(out_dir, side_file))

    meta = {"model": MODEL_NAME, "method": "gptq_fake", "bits": args.bits, "group_size": args.group_size,
            "calibration": {"dataset": "wikitext-2-raw-v1 train", "nsamples": args.nsamples,
                            "seqlen": args.seqlen, "seed": args.seed},
            "percdamp": 0.01, "total_seconds": round(time.time() - start_time), "layers": log,
            "calibration_ppl_2_samples": loss.exp().item()}
    json.dump(meta, open(os.path.join(out_dir, "gptq_meta.json"), "w"), indent=2)
    print(f"\nDone in {time.time() - start_time:.0f}s. Saved {len(weight_map)} tensors to {out_dir}")


if __name__ == "__main__":
    main()
