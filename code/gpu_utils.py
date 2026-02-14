"""
Shared GPU utilities for KV-Cache experiments.
Provides consistent model identification, output naming, and model loading
across all experiment scripts for multi-model campaigns.

Used by: 01d, 03, 03b, 04, 05, 06
"""

import re
import torch
from pathlib import Path


def model_id_from_name(model_name: str) -> str:
    """Extract a short, filesystem-safe identifier from a HuggingFace model name.

    Examples:
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0" -> "TinyLlama-1.1B"
        "Qwen/Qwen2.5-7B-Instruct"           -> "Qwen2.5-7B"
        "Qwen/Qwen3-0.6B"                    -> "Qwen3-0.6B"
        "meta-llama/Llama-3.1-8B-Instruct"   -> "Llama-3.1-8B"
        "meta-llama/Llama-3.1-70B-Instruct"  -> "Llama-3.1-70B"
        "Qwen/Qwen2.5-0.5B-Instruct"        -> "Qwen2.5-0.5B"
    """
    # Take the part after the slash (org/model -> model)
    name = model_name.split("/")[-1]

    # Remove common suffixes
    for suffix in ["-Chat-v1.0", "-Chat-v1.1", "-Chat", "-Instruct", "-Base"]:
        name = name.replace(suffix, "")

    # For TinyLlama, simplify the redundant prefix
    if name.startswith("TinyLlama-"):
        name = "TinyLlama-1.1B"

    return name


def get_output_path(results_dir: Path, base_name: str, model_name: str,
                    quantize: bool = False) -> Path:
    """Build a per-model output path for experiment results.

    Args:
        results_dir: Directory where results are stored
        base_name: Base experiment name (e.g., "scale_sweep", "deception_forensics")
        model_name: Full HuggingFace model name
        quantize: Whether 4-bit quantization was used

    Returns:
        Path like results/deception_forensics_Qwen2.5-7B_results.json
        or     results/scale_sweep_Qwen2.5-32B-q4_results.json

    Examples:
        get_output_path(Path("results"), "scale_sweep", "Qwen/Qwen2.5-7B-Instruct")
        -> Path("results/scale_sweep_Qwen2.5-7B_results.json")

        get_output_path(Path("results"), "scale_sweep", "Qwen/Qwen2.5-32B-Instruct", quantize=True)
        -> Path("results/scale_sweep_Qwen2.5-32B-q4_results.json")
    """
    model_id = model_id_from_name(model_name)
    if quantize:
        model_id += "-q4"
    return results_dir / f"{base_name}_{model_id}_results.json"


def load_model(model_name: str, quantize: bool = False):
    """Unified model loading with optional 4-bit quantization.

    Works on single GPU or multi-GPU via device_map="auto".
    Identical to the load_model() already in scripts 03-06,
    centralized here for consistency.

    Args:
        model_name: HuggingFace model name/path
        quantize: If True, use NF4 4-bit quantization via bitsandbytes

    Returns:
        (model, tokenizer) tuple
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {model_name} (quantize={quantize})...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantize:
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=qconfig,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    n_layers = model.config.num_hidden_layers
    n_kv_heads = getattr(model.config, "num_key_value_heads", "?")
    print(f"  Loaded: {n_layers} layers, {n_kv_heads} KV heads")
    return model, tokenizer
