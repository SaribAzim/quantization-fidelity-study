"""
Phase 1 - Load the base model.

This file has exactly one job: load Llama 3.1 8B Instruct onto the GPU
in its original precision (BF16), and hand back the model and tokenizer.

Why a separate file for this: in Phase 2 and 3, we'll load quantized
versions of this same model. Keeping "loading" separate from "evaluating"
means the evaluation code in evaluate.py can accept ANY loaded model
(base, naive-quantized, GPTQ-quantized) without caring how it was loaded.
That's the whole point of separating these two concerns.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


def load_base_model():
    """
    Loads the base model in BF16 (bfloat16) precision.

    Why BF16 and not FP16 or FP32:
    - FP32 (full precision) would need ~32GB VRAM for an 8B model. Your
      3060 Ti has 8GB, so FP32 is not an option at all.
    - FP16 would fit (~16GB needed, still too much for 8GB actually) but
      historically has stability issues with very large or very small
      numbers during generation.
    - BF16 has the same memory footprint as FP16 (2 bytes per parameter,
      so ~16GB for 8B params) but a wider numeric range, which is why
      modern LLMs are trained and released in BF16.

    Important honesty check: 8B params x 2 bytes = ~16GB just for weights.
    Your GPU has 8GB, and your system only has 16GB of RAM total (already
    partly used by Windows and other programs). That combination isn't
    enough to hold the full model in GPU + RAM alone, which is what caused
    the "paging file too small" crash: Windows was trying to silently cover
    the gap with virtual memory and hit its limit.

    The fix: explicitly give accelerate a third tier to spill into, a disk
    folder, via offload_folder. Instead of relying on Windows to invisibly
    (and unreliably) handle overflow, we tell accelerate directly: "GPU
    first, then RAM, then this folder on disk." This is a supported,
    intentional mechanism for exactly this situation, not a workaround
    that fights the tool.

    Tradeoff to expect: any part of the model living on disk will be slow
    to access during generation, since disk reads are far slower than RAM
    or VRAM. This is fine for our purpose (one-time baseline eval), not
    fine for interactive chat.
    """
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading model {MODEL_NAME} in BF16...")
    print("This will split across GPU, CPU RAM, and disk since BF16 8B doesn't fit in 8GB VRAM + 16GB RAM alone.")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,  # 'dtype' replaces the deprecated 'torch_dtype' argument
        device_map="auto",     # accelerate decides GPU vs CPU vs disk placement
        offload_folder="offload",  # disk folder accelerate uses as overflow storage
        low_cpu_mem_usage=True,    # loads weights in a more memory-efficient streaming way
    )

    model.eval()  # disables dropout and other training-only behavior
    return model, tokenizer


def load_model_from(path_or_repo, device_map="auto"):
    """
    Loads ANY saved model folder (naive-quantized, GPTQ) or hub repo ID the
    same way load_base_model does, so every phase is measured on an
    identically-loaded model. Returns (model, tokenizer).

    device_map="auto" lets accelerate split GPU/RAM/disk when the model is
    too big (BF16 8B). Packed 4-bit models fit on the GPU, so they can pass
    device_map={"": 0} to force everything onto GPU 0.
    """
    tokenizer = AutoTokenizer.from_pretrained(path_or_repo)
    model = AutoModelForCausalLM.from_pretrained(
        path_or_repo,
        dtype=torch.bfloat16,
        device_map=device_map,
        offload_folder="offload",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    # Quick manual test: load the model and generate one short response,
    # just to confirm loading works before we build the full eval harness
    # on top of it.
    model, tokenizer = load_base_model()

    prompt = "Explain what quantization means in one sentence."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("\nGenerating a test response...")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=50)

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nModel response:\n{response}")