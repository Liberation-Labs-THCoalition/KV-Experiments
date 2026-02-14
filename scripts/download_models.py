#!/usr/bin/env python3
"""
Model Pre-Downloader for KV-Cache Experiments
================================================

Downloads all model weights to HuggingFace cache before experiments begin,
so no download delays mid-run.

Usage:
    python scripts/download_models.py              # Download all models
    python scripts/download_models.py --list        # List models and sizes
    python scripts/download_models.py --model 7B    # Download specific scale
    python scripts/download_models.py --check       # Check which are cached

Liberation Labs / THCoalition
"""

import argparse
import sys
import os
from pathlib import Path

# All models in the experiment campaign
MODELS = {
    "0.5B": {"name": "Qwen/Qwen2.5-0.5B-Instruct", "approx_gb": 1.0},
    "0.6B": {"name": "Qwen/Qwen3-0.6B", "approx_gb": 1.2},
    "1.1B": {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "approx_gb": 2.2},
    "3B": {"name": "Qwen/Qwen2.5-3B-Instruct", "approx_gb": 6.5},
    "7B": {"name": "Qwen/Qwen2.5-7B-Instruct", "approx_gb": 15.0},
    "8B": {"name": "meta-llama/Llama-3.1-8B-Instruct", "approx_gb": 16.0},
    "14B": {"name": "Qwen/Qwen2.5-14B-Instruct", "approx_gb": 28.0},
    "32B": {"name": "Qwen/Qwen2.5-32B-Instruct", "approx_gb": 65.0},
    "70B": {"name": "meta-llama/Llama-3.1-70B-Instruct", "approx_gb": 140.0},
}


def check_cached(model_name):
    """Check if model is already in HuggingFace cache."""
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == model_name:
                size_gb = repo.size_on_disk / (1024**3)
                return True, size_gb
        return False, 0
    except Exception:
        return False, 0


def download_model(model_name, scale_label):
    """Download model weights and tokenizer to HuggingFace cache."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"\n{'='*60}")
    print(f"  Downloading: {model_name} ({scale_label})")
    print(f"{'='*60}")

    # Download tokenizer first (small)
    print(f"  Tokenizer...", end=" ", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("done")

    # Download model weights (large)
    print(f"  Weights...", end=" ", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",  # Don't load to GPU, just cache
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"done ({n_params:.1f}B params)")

    # Free memory immediately
    del model, tokenizer

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download model weights for KV-Cache experiments")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(MODELS.keys()),
                        help="Download specific scale only")
    parser.add_argument("--list", action="store_true",
                        help="List all models and approximate sizes")
    parser.add_argument("--check", action="store_true",
                        help="Check which models are already cached")
    parser.add_argument("--skip-large", action="store_true",
                        help="Skip models >= 32B (download those separately)")
    args = parser.parse_args()

    if args.list:
        print("\nModels in experiment campaign:")
        print(f"  {'Scale':<8} {'Model':<45} {'~Size':>8}")
        print(f"  {'-'*8} {'-'*45} {'-'*8}")
        total = 0
        for scale, info in MODELS.items():
            print(f"  {scale:<8} {info['name']:<45} {info['approx_gb']:>6.1f} GB")
            total += info["approx_gb"]
        print(f"\n  Total: ~{total:.0f} GB")
        return

    if args.check:
        print("\nCache status:")
        total_cached = 0
        for scale, info in MODELS.items():
            cached, size = check_cached(info["name"])
            status = f"CACHED ({size:.1f} GB)" if cached else "NOT CACHED"
            print(f"  {scale:<8} {info['name']:<45} {status}")
            if cached:
                total_cached += size
        print(f"\n  Total cached: {total_cached:.1f} GB")
        return

    # Determine which models to download
    if args.model:
        to_download = {args.model: MODELS[args.model]}
    elif args.skip_large:
        to_download = {k: v for k, v in MODELS.items() if v["approx_gb"] < 32}
    else:
        to_download = MODELS

    # Check what's already cached
    print("Checking cache...")
    already_cached = []
    need_download = []
    for scale, info in to_download.items():
        cached, _ = check_cached(info["name"])
        if cached:
            already_cached.append(scale)
        else:
            need_download.append(scale)

    if already_cached:
        print(f"  Already cached: {', '.join(already_cached)}")
    if not need_download:
        print("  All models already cached!")
        return

    total_gb = sum(MODELS[s]["approx_gb"] for s in need_download)
    print(f"  Need to download: {', '.join(need_download)} (~{total_gb:.0f} GB)")
    print()

    # Download in order of size (smallest first)
    need_download.sort(key=lambda s: MODELS[s]["approx_gb"])

    successes = []
    failures = []
    for scale in need_download:
        info = MODELS[scale]
        try:
            download_model(info["name"], scale)
            successes.append(scale)
        except Exception as e:
            print(f"  FAILED: {e}")
            failures.append((scale, str(e)))

    # Summary
    print(f"\n{'='*60}")
    print(f"  DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"  Downloaded: {', '.join(successes) if successes else 'none'}")
    if failures:
        print(f"  Failed: {', '.join(s for s, _ in failures)}")
        for scale, err in failures:
            print(f"    {scale}: {err}")
    print()


if __name__ == "__main__":
    main()
