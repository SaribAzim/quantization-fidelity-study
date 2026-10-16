"""
REAL 4-bit (and 8-bit) storage, instead of "fake" quantization.

Until now we rounded the weights and wrote them back as 16-bit numbers, so a
model was still 15 GB and could not fit on the GPU. Here we store what real
quantization stores:
    qweight   the small integers, two 4-bit values packed into every byte
    scales    one number per row (or per group of columns) to turn integers
              back into weights
    zeros     the matching zero-points
The model drops to about 5.3 GB and fits on the 8 GB GPU.

At run time each layer unpacks its weights just before using them (plain
PyTorch, no special compiler needed) with exactly the same arithmetic as the
fake-quantized version, so the numbers coming out are identical.

LoRA (the "L" in QLoRA): a layer can also carry two small trainable matrices
A and B. Its output becomes  base_output + (x @ A^T) @ B^T * scaling.  B
starts at zero, so a fresh adapter changes nothing; training then teaches A
and B to correct the rounding damage while the 4-bit weights stay frozen.
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def pack_integers(q, bits):
    """q: (rows, cols) integers in [0, 2**bits). 4-bit: two columns per byte (even column low, odd column high)."""
    q = q.to(torch.uint8)
    if bits == 8:
        return q
    assert bits == 4 and q.size(1) % 2 == 0, "only 4 and 8 bits are supported (and 4-bit needs an even width)"
    return q[:, 0::2] | (q[:, 1::2] << 4)


def unpack_integers(packed, bits, cols):
    if bits == 8:
        return packed
    low, high = packed & 0x0F, packed >> 4
    return torch.stack([low, high], dim=-1).reshape(packed.size(0), cols)


class PackedLinear(nn.Module):
    """A Linear layer whose frozen weights live as packed integers (see the module docstring)."""

    def __init__(self, in_features, out_features, bits=4, group_size=-1, device=None):
        super().__init__()
        self.in_features, self.out_features, self.bits, self.group_size = in_features, out_features, bits, group_size
        groups = 1 if group_size == -1 else in_features // group_size
        self.register_buffer("qweight", torch.zeros(out_features, in_features * bits // 8, dtype=torch.uint8, device=device))
        self.register_buffer("scales", torch.zeros(out_features, groups, dtype=torch.float32, device=device))
        self.register_buffer("zeros", torch.zeros(out_features, groups, dtype=torch.float32, device=device))
        self.lora_A = None
        self.lora_B = None
        self.lora_scaling = 1.0

    @classmethod
    def from_integers(cls, quantized, scales, zero_points, bits, group_size):
        """Builds a layer from the output of quantize_tensor_to_integers(_grouped)."""
        out_features, in_features = quantized.shape
        layer = cls(in_features, out_features, bits, group_size, device=quantized.device)
        layer.qweight.copy_(pack_integers(quantized, bits))
        layer.scales.copy_(scales.reshape(out_features, -1))
        layer.zeros.copy_(zero_points.reshape(out_features, -1))
        return layer

    def dequantize(self):
        """Back to a BF16 weight matrix, using the same arithmetic as fake quantization (float32, then BF16)."""
        q = unpack_integers(self.qweight, self.bits, self.in_features).float()
        groups = self.scales.size(1)
        q = q.view(self.out_features, groups, self.in_features // groups)
        weight = (q - self.zeros.unsqueeze(-1)) * self.scales.unsqueeze(-1)
        return weight.view(self.out_features, self.in_features).to(torch.bfloat16)

    def add_lora(self, rank, alpha, device):
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)  # standard LoRA init; B = 0 means "no change yet"
        self.lora_scaling = alpha / rank

    def forward(self, x):
        y = F.linear(x, self.dequantize())
        if self.lora_A is not None:
            y = y + F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype)) * self.lora_scaling
        return y


class CpuEmbedding(nn.Module):
    """Keeps the 1 GB word-embedding table in CPU RAM (it is only looked up, never trained), freeing GPU memory."""

    def __init__(self, weight, device):
        super().__init__()
        self.weight_cpu, self.device = weight.cpu(), device

    def forward(self, ids):
        return F.embedding(ids.cpu(), self.weight_cpu).to(self.device)


def load_packed_model(folder, device="cuda", embeddings_on_cpu=False):
    """
    Builds the model skeleton without allocating memory, swaps the 224 decoder
    Linear layers for PackedLinear, allocates the (small) packed sizes on the
    GPU, and fills everything from the packed files. Returns (model, tokenizer).
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    from quantization.quantize_checkpoint import is_quantizable
    from quantization.run_gptq import CheckpointReader

    config = AutoConfig.from_pretrained(folder)
    settings = json.load(open(os.path.join(folder, "packed_config.json")))
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and is_quantizable(name + ".weight"):
            bits, group_size = settings["overrides"].get(name, [settings["bits"], settings["group_size"]])
            parent_name, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent_name), child,
                    PackedLinear(module.in_features, module.out_features, bits, group_size, device="meta"))

    model.to_empty(device=device)
    # to_empty leaves non-saved buffers uninitialized; the rotary table is one of them, so rebuild it.
    model.model.rotary_emb = LlamaRotaryEmbedding(config=config, device=device)

    reader = CheckpointReader(folder)
    for key, tensor in model.state_dict().items():
        tensor.copy_(reader.get(key))
    for p in model.parameters():
        p.requires_grad_(False)
    if embeddings_on_cpu:
        model.model.embed_tokens = CpuEmbedding(model.model.embed_tokens.weight.data, device)
    model.eval()
    return model, AutoTokenizer.from_pretrained(folder)


def add_lora_to_model(model, rank=16, alpha=32, device="cuda"):
    """Attaches trainable LoRA matrices to all 224 packed layers; returns the list of trainable parameters."""
    params = []
    for module in model.modules():
        if isinstance(module, PackedLinear):
            module.add_lora(rank, alpha, device)
            params += [module.lora_A, module.lora_B]
    return params


def lora_state(model):
    state = {f"{name}.{p}": getattr(m, p).detach().cpu() for name, m in model.named_modules()
             if isinstance(m, PackedLinear) and m.lora_A is not None for p in ("lora_A", "lora_B")}
    first = next(m for m in model.modules() if isinstance(m, PackedLinear) and m.lora_A is not None)
    state["alpha"] = torch.tensor(float(first.lora_scaling * first.lora_A.size(0)))  # rank * scaling = alpha
    return state


def load_lora_state(model, state, device="cuda"):
    for name, m in model.named_modules():
        if isinstance(m, PackedLinear) and f"{name}.lora_A" in state:
            rank = state[f"{name}.lora_A"].size(0)
            m.add_lora(rank, state["alpha"].item() if "alpha" in state else 2 * rank, device)
            m.lora_A.data.copy_(state[f"{name}.lora_A"])
            m.lora_B.data.copy_(state[f"{name}.lora_B"])
