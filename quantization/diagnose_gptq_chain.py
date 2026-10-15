"""
Diagnostic 2: which GPTQ settings actually lower the END-TO-END loss?

Layer-output error looked fine for our GPTQ, yet the full model came out worse
than plain rounding. Layer error is dominated by a few huge "massive
activation" values (Llama puts giant numbers on the first token of a window),
so it can hide damage to everything else. The honest measure is the final
language-modelling loss on HELD-OUT text.

For each variant we quantize only the first N decoder layers (sequentially,
exactly as run_gptq.py does), leave the rest full precision, and measure the
final held-out loss. Same layers, same text: only the method differs.

Run: python quantization/diagnose_gptq_chain.py --nlayers 6
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantization.gptq import HessianAccumulator, gptq_quantize
from quantization.naive_ptq import quantize_tensor_per_group
from quantization.quantize_checkpoint import find_cached_snapshot
from quantization.run_gptq import (DEVICE, LAYER_LINEARS, CheckpointReader, build_layer,
                                   load_calibration_ids, run_layer)

# name -> (method, percdamp, skip the first token of each window when building H)
VARIANTS = {
    "RTN g128": ("rtn", None, False),
    "GPTQ damp0.01": ("gptq", 0.01, False),
    "GPTQ damp0.10": ("gptq", 0.10, False),
    "GPTQ damp0.01 + skip first token": ("gptq", 0.01, True),
    "GPTQ damp0.10 + skip first token": ("gptq", 0.10, True),
}


@torch.no_grad()
def heldout_loss(final_hidden, ids, reader, config):
    norm_w = reader.get("model.norm.weight").to(DEVICE).float()
    lm_head = reader.get("lm_head.weight").to(DEVICE)
    h = final_hidden.to(DEVICE).float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + config.rms_norm_eps) * norm_w
    losses = []
    for i in range(h.size(0)):
        logits = (h[i:i + 1].to(torch.bfloat16) @ lm_head.T).float()
        losses.append(F.cross_entropy(logits[0, :-1], ids[i, 1:].to(DEVICE)).item())
    return sum(losses) / len(losses)


@torch.no_grad()
def run_variant(name, spec, quantize_layers, args, reader, config, ids, base_inputs, position_ids, pe):
    method, damp, skip_first = spec
    n_fit = args.nfit
    inputs = base_inputs.clone()
    outputs = torch.zeros_like(inputs)

    for layer_idx in range(config.num_hidden_layers):
        layer = build_layer(config, reader, layer_idx)
        if method is not None and layer_idx < quantize_layers:
            modules = {n: layer.get_submodule(n) for n in LAYER_LINEARS}
            if method == "gptq":
                accs, hooks = {}, []
                for n, key in LAYER_LINEARS.items():
                    if key not in accs:
                        accs[key] = HessianAccumulator(modules[n].in_features, DEVICE)
                        first = 1 if skip_first else 0
                        hooks.append(modules[n].register_forward_hook(
                            lambda m, i, o, a=accs[key], f=first: a.add_batch(i[0][:, f:])))
                run_layer(layer, inputs[:n_fit], outputs[:n_fit], 4, position_ids, pe)
                for h in hooks:
                    h.remove()
            for n, key in LAYER_LINEARS.items():
                w = modules[n].weight.data
                if method == "rtn":
                    q = quantize_tensor_per_group(w.float(), args.bits, args.group_size)
                else:
                    q = gptq_quantize(w.float(), accs[key].H, args.bits, args.group_size, percdamp=damp)
                modules[n].weight.data = q.to(w.dtype)
            if method == "gptq":
                del accs
        run_layer(layer, inputs, outputs, 4, position_ids, pe)
        inputs, outputs = outputs, inputs
        del layer
        torch.cuda.empty_cache()

        if (layer_idx + 1) in args.checkpoints and layer_idx + 1 < config.num_hidden_layers:
            # Branch: finish the remaining layers at FULL precision on the held-out
            # samples only, to see the loss if quantization stopped right here.
            branch_in, branch_out = inputs[n_fit:].clone(), torch.zeros_like(inputs[n_fit:])
            for j in range(layer_idx + 1, config.num_hidden_layers):
                fp_layer = build_layer(config, reader, j)
                run_layer(fp_layer, branch_in, branch_out, 4, position_ids, pe)
                branch_in, branch_out = branch_out, branch_in
                del fp_layer
            loss = heldout_loss(branch_in, ids[n_fit:], reader, config)
            print(f"CHECKPOINT {name:20s} first {layer_idx + 1:2d} layers quantized: loss {loss:.4f}", flush=True)

    return heldout_loss(inputs[n_fit:], ids[n_fit:], reader, config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlayers", type=int, default=6, help="how many of the first layers to quantize")
    parser.add_argument("--nsamples", type=int, default=48)
    parser.add_argument("--nfit", type=int, default=32, help="samples used for Hessians; the rest are held out")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--only", nargs="*", default=None, help="subset of variant names")
    parser.add_argument("--checkpoints", type=int, nargs="*", default=[], help="report loss after this many quantized layers")
    args = parser.parse_args()

    src_dir = find_cached_snapshot()
    reader = CheckpointReader(src_dir)
    config = AutoConfig.from_pretrained(src_dir)
    config._attn_implementation = "sdpa"
    ids = load_calibration_ids(AutoTokenizer.from_pretrained(src_dir), args.nsamples, args.seqlen, 0)

    embed = reader.get("model.embed_tokens.weight").to(DEVICE)
    base_inputs = torch.stack([F.embedding(ids[i].to(DEVICE), embed).cpu() for i in range(args.nsamples)])
    del embed
    rotary = LlamaRotaryEmbedding(config=config).to(DEVICE)
    position_ids = torch.arange(args.seqlen, device=DEVICE).unsqueeze(0)
    pe = rotary(base_inputs[:1].to(DEVICE), position_ids)

    results = {}
    fp = run_variant("FP", (None, None, False), 0, args, reader, config, ids, base_inputs, position_ids, pe)
    print(f"full precision held-out loss: {fp:.4f}  (ppl {torch.tensor(fp).exp():.2f})", flush=True)
    for name, spec in VARIANTS.items():
        if args.only and name not in args.only:
            continue
        loss = run_variant(name, spec, args.nlayers, args, reader, config, ids, base_inputs, position_ids, pe)
        results[name] = loss
        print(f"RESULT {name:36s} loss {loss:.4f}  (+{loss - fp:.4f} vs FP)  ppl {torch.tensor(loss).exp():.2f}", flush=True)


if __name__ == "__main__":
    main()
