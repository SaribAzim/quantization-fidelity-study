"""
Diagnostic: is our GPTQ actually better than plain rounding, layer by layer?

For a few chosen decoder layers we take the full-precision inputs, collect
Hessians, then quantize the layer three ways and measure how far the layer's
OUTPUT moves on HELD-OUT text (text not used to build the Hessians):
    RTN g128        plain rounding, same grid as GPTQ
    GPTQ damp 0.01  what run_gptq.py does
    GPTQ damp 0.10  a more heavily damped (more stable) Hessian inverse
Lower relative error is better. GPTQ should clearly beat RTN.

Run: python quantization/diagnose_gptq.py --layers 0 8 31
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


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 8, 31])
    parser.add_argument("--nsamples", type=int, default=48, help="48 = 32 for Hessians + 16 held out")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()
    n_fit = args.nsamples * 2 // 3

    src_dir = find_cached_snapshot()
    reader = CheckpointReader(src_dir)
    config = AutoConfig.from_pretrained(src_dir)
    config._attn_implementation = "sdpa"
    ids = load_calibration_ids(AutoTokenizer.from_pretrained(src_dir), args.nsamples, args.seqlen, 0)

    embed = reader.get("model.embed_tokens.weight").to(DEVICE)
    inputs = torch.stack([F.embedding(ids[i].to(DEVICE), embed).cpu() for i in range(args.nsamples)])
    del embed
    outputs = torch.zeros_like(inputs)
    rotary = LlamaRotaryEmbedding(config=config).to(DEVICE)
    position_ids = torch.arange(args.seqlen, device=DEVICE).unsqueeze(0)
    pe = rotary(inputs[:1].to(DEVICE), position_ids)

    for layer_idx in range(max(args.layers) + 1):
        layer = build_layer(config, reader, layer_idx)
        if layer_idx in args.layers:
            modules = {n: layer.get_submodule(n) for n in LAYER_LINEARS}
            accs, hooks = {}, []
            for n, key in LAYER_LINEARS.items():
                if key not in accs:
                    accs[key] = HessianAccumulator(modules[n].in_features, DEVICE)
                    hooks.append(modules[n].register_forward_hook(lambda m, i, o, a=accs[key]: a.add_batch(i[0])))
            run_layer(layer, inputs[:n_fit], outputs[:n_fit], 4, position_ids, pe)  # fit Hessians
            for h in hooks:
                h.remove()

            held_in, fp_out = inputs[n_fit:], torch.zeros_like(inputs[n_fit:])
            run_layer(layer, held_in, fp_out, 4, position_ids, pe)
            originals = {n: modules[n].weight.data.clone() for n in LAYER_LINEARS}

            def evaluate(make_q):
                for n, key in LAYER_LINEARS.items():
                    modules[n].weight.data = make_q(originals[n].float(), accs[key].H).to(originals[n].dtype)
                out = torch.zeros_like(held_in)
                run_layer(layer, held_in, out, 4, position_ids, pe)
                for n in LAYER_LINEARS:
                    modules[n].weight.data = originals[n].clone()
                return ((out.float() - fp_out.float()).norm() / fp_out.float().norm()).item()

            rtn = evaluate(lambda w, H: quantize_tensor_per_group(w, args.bits, args.group_size))
            g01 = evaluate(lambda w, H: gptq_quantize(w, H, args.bits, args.group_size, percdamp=0.01))
            g10 = evaluate(lambda w, H: gptq_quantize(w, H, args.bits, args.group_size, percdamp=0.10))
            print(f"layer {layer_idx:2d} held-out relative output error:  RTN {rtn:.4f} | GPTQ damp0.01 {g01:.4f} | GPTQ damp0.10 {g10:.4f}", flush=True)
            del accs

        run_layer(layer, inputs, outputs, 4, position_ids, pe)  # FP chain to the next layer
        inputs, outputs = outputs, inputs
        del layer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
