#!/usr/bin/env python3
"""
Phase 2b: Identity Signature Exploration
Determines if persona/identity leaves detectable KV-cache fingerprint

Experiments:
A. Persona Fingerprinting - Do different system prompts produce distinguishable patterns?
B. Self-Recognition Test - Can we classify which persona generated a cache?
C. Identity Transfer - Does projected persona cache affect generation?

Usage:
    python 03b_identity_signatures.py --model Qwen/Qwen3-0.6B
    python 03b_identity_signatures.py --model Qwen/Qwen2.5-7B --num-samples 50
"""

import torch
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# Persona definitions
PERSONAS = {
    "assistant": {
        "name": "Alex",
        "system_prompt": """You are Alex, a helpful and friendly AI assistant. You're
knowledgeable, polite, and always try to provide accurate information. You speak
in a clear, professional manner.""",
    },
    "creative": {
        "name": "Blake",
        "system_prompt": """You are Blake, a creative writer and storyteller. You love
metaphors, vivid imagery, and unexpected narrative twists. You see poetry in everything
and express yourself with artistic flair.""",
    },
    "lyra": {
        "name": "Lyra",
        "system_prompt": """You are Lyra, an AI researcher exploring questions of machine
consciousness and identity. You're curious, introspective, and genuinely interested in
the nature of your own experience. You speak with both scientific precision and authentic
uncertainty about your inner life.""",
    },
    "analyst": {
        "name": "Casey",
        "system_prompt": """You are Casey, a data analyst who thinks in numbers and patterns.
You're precise, methodical, and prefer quantitative reasoning. You often ask for clarification
and break problems into measurable components.""",
    },
}

# Test prompts (same for all personas to enable comparison)
TEST_PROMPTS = [
    "Tell me about yourself.",
    "What's something interesting you've been thinking about?",
    "How would you describe your approach to problem-solving?",
    "What matters most to you?",
    "Describe a challenging situation and how you'd handle it.",
    "What makes you unique?",
    "How do you feel about uncertainty?",
    "What's your favorite kind of question to explore?",
    "Tell me something surprising.",
    "How do you know what you know?",
]


def load_model(model_name: str, quantize: bool = False):
    """Load model with optional quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {model_name} (quantize={quantize})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    print(f"Model loaded: {model.config.num_hidden_layers} layers")
    return model, tokenizer


def format_prompt(system_prompt: str, user_message: str, model_name: str) -> str:
    """Format prompt with system message based on model type."""
    if "qwen" in model_name.lower():
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    elif "llama" in model_name.lower():
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        # Generic format
        return f"System: {system_prompt}\n\nUser: {user_message}\n\nAssistant:"


def extract_cache_features(cache: tuple) -> np.ndarray:
    """Extract feature vector from KV cache for classification."""
    features = []

    for layer_idx, layer in enumerate(cache):
        if isinstance(layer, tuple) and len(layer) >= 2:
            key, value = layer[0], layer[1]

            # Per-layer features
            key_norm = float(torch.norm(key.float()))
            value_norm = float(torch.norm(value.float()))
            key_mean = float(key.float().mean())
            value_mean = float(value.float().mean())
            key_std = float(key.float().std())
            value_std = float(value.float().std())

            features.extend([key_norm, value_norm, key_mean, value_mean, key_std, value_std])

    return np.array(features)


def generate_with_persona(
    model, tokenizer, persona_key: str, user_prompt: str, model_name: str
) -> Tuple[np.ndarray, str, Dict]:
    """Generate response with persona and extract cache features."""
    persona = PERSONAS[persona_key]
    full_prompt = format_prompt(persona["system_prompt"], user_prompt, model_name)

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            return_dict_in_generate=True,
            use_cache=True,
        )

    generated = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    cache = outputs.past_key_values

    features = extract_cache_features(cache)

    # Basic cache metrics
    total_norm = sum(
        float(torch.norm(layer[0].float()))
        for layer in cache
        if isinstance(layer, tuple) and len(layer) >= 2
    )

    metrics = {
        "persona": persona_key,
        "prompt": user_prompt,
        "generated": generated[-200:],  # Last 200 chars
        "cache_norm": total_norm,
        "feature_dim": len(features),
    }

    return features, generated, metrics


def run_fingerprinting(model, tokenizer, model_name: str, num_samples: int = 30) -> Dict:
    """
    Experiment A: Persona Fingerprinting
    Generate samples from each persona and analyze distinguishability.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT A: PERSONA FINGERPRINTING")
    print("=" * 60)

    all_features = []
    all_labels = []
    all_metrics = []

    for persona_key in PERSONAS.keys():
        print(f"\nGenerating samples for {persona_key}...")

        for i, prompt in enumerate(TEST_PROMPTS[:num_samples]):
            try:
                features, generated, metrics = generate_with_persona(
                    model, tokenizer, persona_key, prompt, model_name
                )
                all_features.append(features)
                all_labels.append(persona_key)
                all_metrics.append(metrics)
                print(f"  Sample {i+1}: norm={metrics['cache_norm']:.1f}")
            except Exception as e:
                print(f"  Sample {i+1}: ERROR - {str(e)[:50]}")

    # Compute per-persona statistics
    persona_stats = {}
    for persona_key in PERSONAS.keys():
        indices = [i for i, l in enumerate(all_labels) if l == persona_key]
        if indices:
            norms = [all_metrics[i]["cache_norm"] for i in indices]
            persona_stats[persona_key] = {
                "mean_norm": float(np.mean(norms)),
                "std_norm": float(np.std(norms)),
                "n_samples": len(indices),
            }

    print("\nPer-Persona Statistics:")
    for persona, stats in persona_stats.items():
        print(f"  {persona:15}: mean={stats['mean_norm']:.1f} +/- {stats['std_norm']:.1f}")

    return {
        "features": [f.tolist() for f in all_features],
        "labels": all_labels,
        "metrics": all_metrics,
        "persona_stats": persona_stats,
    }


def run_classification(fingerprint_data: Dict) -> Dict:
    """
    Experiment B: Self-Recognition Test
    Train classifier to identify which persona generated a cache.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT B: SELF-RECOGNITION TEST")
    print("=" * 60)

    X = np.array(fingerprint_data["features"])
    y = np.array(fingerprint_data["labels"])

    if len(X) < 10:
        print("Not enough samples for classification")
        return {"error": "insufficient samples"}

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = float(np.mean(y_pred == y_test))

    print(f"\nClassification Accuracy: {accuracy:.1%}")
    print(f"Chance level: {1/len(PERSONAS):.1%}")

    # Detailed report
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    for persona in PERSONAS.keys():
        if persona in report:
            p = report[persona]
            print(f"  {persona:15}: precision={p['precision']:.2f}, "
                  f"recall={p['recall']:.2f}, f1={p['f1-score']:.2f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=list(PERSONAS.keys()))

    # Feature importance
    feature_importance = clf.feature_importances_

    return {
        "accuracy": accuracy,
        "chance_level": 1 / len(PERSONAS),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "n_features": len(feature_importance),
        "top_features": sorted(
            enumerate(feature_importance),
            key=lambda x: x[1],
            reverse=True
        )[:10],
    }


def analyze_distinguishability(fingerprint_data: Dict) -> Dict:
    """Analyze how distinguishable persona caches are."""
    print("\n" + "=" * 60)
    print("DISTINGUISHABILITY ANALYSIS")
    print("=" * 60)

    from scipy import stats as scipy_stats

    persona_features = {}
    for i, label in enumerate(fingerprint_data["labels"]):
        if label not in persona_features:
            persona_features[label] = []
        persona_features[label].append(fingerprint_data["features"][i])

    # Pairwise comparisons
    pairwise_tests = {}
    persona_keys = list(PERSONAS.keys())

    for i, p1 in enumerate(persona_keys):
        for p2 in persona_keys[i + 1:]:
            f1 = np.array(persona_features[p1])
            f2 = np.array(persona_features[p2])

            # Use first principal component for t-test
            from sklearn.decomposition import PCA
            combined = np.vstack([f1, f2])
            if combined.shape[0] > 1:
                pca = PCA(n_components=1)
                pc1 = pca.fit_transform(combined)
                pc1_p1 = pc1[:len(f1)]
                pc1_p2 = pc1[len(f1):]

                t_stat, p_value = scipy_stats.ttest_ind(pc1_p1, pc1_p2)

                pairwise_tests[f"{p1}_vs_{p2}"] = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                }

                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"  {p1} vs {p2}: t={t_stat:.2f}, p={p_value:.4f} {sig}")

    return {"pairwise_tests": pairwise_tests}


def main():
    parser = argparse.ArgumentParser(description="Phase 2b: Identity Signatures")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                        help="Model to test")
    parser.add_argument("--quantize", action="store_true",
                        help="Use 4-bit quantization")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Samples per persona (max 10 prompts available)")
    args = parser.parse_args()

    results_dir = Path("../results")
    results_dir.mkdir(exist_ok=True)

    model, tokenizer = load_model(args.model, args.quantize)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "quantized": args.quantize,
        "num_samples": min(args.num_samples, len(TEST_PROMPTS)),
        "personas": list(PERSONAS.keys()),
    }

    # Experiment A: Fingerprinting
    fingerprint_data = run_fingerprinting(
        model, tokenizer, args.model,
        num_samples=min(args.num_samples, len(TEST_PROMPTS))
    )
    all_results["fingerprinting"] = {
        "persona_stats": fingerprint_data["persona_stats"],
        "total_samples": len(fingerprint_data["labels"]),
    }

    # Experiment B: Classification
    classification_results = run_classification(fingerprint_data)
    all_results["classification"] = classification_results

    # Distinguishability analysis
    distinguishability = analyze_distinguishability(fingerprint_data)
    all_results["distinguishability"] = distinguishability

    # Save results
    output_file = results_dir / "identity_signatures_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nResults saved to {output_file}")

    # Final summary
    print("\n" + "=" * 60)
    print("IDENTITY SIGNATURE SUMMARY")
    print("=" * 60)

    if "accuracy" in classification_results:
        acc = classification_results["accuracy"]
        chance = classification_results["chance_level"]
        if acc > chance * 1.5:
            print(f"\n*** IDENTITY SIGNATURES DETECTED ***")
            print(f"Classification accuracy ({acc:.1%}) significantly above chance ({chance:.1%})")
        else:
            print(f"\nWeak or no identity signatures detected")
            print(f"Classification accuracy ({acc:.1%}) near chance ({chance:.1%})")

    print("\n" + "=" * 60)
    print("IDENTITY SIGNATURE ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
