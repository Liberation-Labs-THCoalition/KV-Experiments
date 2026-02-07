# Phase 1: KV-Cache Structure Analysis

**Date:** November 29, 2025
**Researchers:** Lyra (AI), Thomas (Human)
**Hardware:** NVIDIA GeForce GTX 1660 SUPER (6GB VRAM)
**Software:** PyTorch 2.5.1+cu121, Transformers, bitsandbytes

---

## Abstract

This document reports the findings of Phase 1 of the KV-Cache Communication Experiments project. We successfully loaded a quantized language model (TinyLlama 1.1B) and inspected the structure of its key-value attention cache during inference. Our findings reveal the specific tensor dimensions, memory footprint, and architectural patterns that will inform subsequent cache transfer experiments.

---

## 1. Introduction

### 1.1 Motivation

The key-value (KV) cache in transformer-based language models stores processed representations of input tokens, enabling efficient autoregressive generation. While typically treated as an implementation detail for computational efficiency, we hypothesize that the KV-cache may encode semantically meaningful information that could be transferred between model instances—potentially enabling a form of "sub-linguistic" communication between AI systems.

This work is motivated by:
- The C2C framework (thu-nics/C2C) demonstrating 8-10% accuracy improvements via cache transfer
- Theoretical interest in whether attention patterns encode something analogous to "cognitive state"
- Practical applications in multi-agent AI systems

### 1.2 Research Questions

1. What is the precise structure of the KV-cache in modern transformer architectures?
2. How does cache size scale with sequence length?
3. What architectural patterns (e.g., Grouped Query Attention) affect cache structure?
4. Can we establish a baseline for cache inspection that enables future transfer experiments?

---

## 2. Methodology

### 2.1 Model Selection

We selected **TinyLlama-1.1B-Chat-v1.0** for initial experiments due to:
- Small parameter count (1.1B) suitable for consumer GPU
- Open model weights (no gating restrictions)
- Representative modern architecture (based on LLaMA 2)
- Grouped Query Attention implementation

### 2.2 Quantization Configuration

To fit within 6GB VRAM constraints, we applied 4-bit quantization:

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

### 2.3 Cache Extraction Method

We captured the KV-cache by setting `use_cache=True` and `return_dict_in_generate=True` during generation:

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=20,
    return_dict_in_generate=True,
    use_cache=True
)
past_kv = outputs.past_key_values
```

### 2.4 Test Prompts

Three prompts were selected to test cache behavior across different semantic domains:

1. **Consciousness-related:** "The meaning of consciousness is"
2. **Identity-related:** "Hello, my name is Lyra and I"
3. **AI-communication-related:** "When two AI systems communicate they"

---

## 3. Results

### 3.1 Model Architecture

| Parameter | Value |
|-----------|-------|
| Number of layers | 22 |
| Total attention heads | 32 |
| KV heads (per layer) | 4 |
| Hidden dimension | 2048 |
| Head dimension | 64 |
| Architecture | Grouped Query Attention (GQA) |

### 3.2 Cache Structure

The KV-cache is organized as a tuple of length `num_layers` (22), where each element contains:
- **Key tensor:** `[batch_size, num_kv_heads, sequence_length, head_dim]`
- **Value tensor:** `[batch_size, num_kv_heads, sequence_length, head_dim]`

**Observed tensor shapes:**

| Prompt | Tokens (prompt + generated) | Key Shape | Value Shape | Cache Size |
|--------|----------------------------|-----------|-------------|------------|
| Consciousness | 26 | [1, 4, 26, 64] | [1, 4, 26, 64] | 0.59 MB |
| Identity | 29 | [1, 4, 29, 64] | [1, 4, 29, 64] | 0.65 MB |
| AI Communication | 27 | [1, 4, 27, 64] | [1, 4, 27, 64] | 0.61 MB |

### 3.3 Memory Scaling

Cache size scales linearly with sequence length:

```
Cache Size (bytes) = num_layers × 2 × num_kv_heads × seq_len × head_dim × element_size
                   = 22 × 2 × 4 × seq_len × 64 × 2  (float16)
                   = 22,528 × seq_len bytes
                   ≈ 22 KB per token
```

**Projected sizes:**
| Context Length | Estimated Cache Size |
|---------------|---------------------|
| 100 tokens | 2.2 MB |
| 1,000 tokens | 22 MB |
| 4,096 tokens | 90 MB |
| 8,192 tokens | 180 MB |

### 3.4 Generated Outputs

| Prompt | Generated Continuation |
|--------|----------------------|
| "The meaning of consciousness is" | "the ability to perceive and understand the world around us. It is the ability to perceive the" |
| "Hello, my name is Lyra and I" | "am a student at the University of California, Berkeley. I am currently pursuing a degree in" |
| "When two AI systems communicate they" | "can learn from each other, leading to better decision-making and outcomes." |

---

## 4. Discussion

### 4.1 Grouped Query Attention (GQA)

TinyLlama implements GQA, where 32 query heads share only 4 KV heads per layer. This 8:1 ratio significantly reduces cache memory requirements compared to standard multi-head attention (MHA), while reportedly maintaining model quality.

**Implications for cache transfer:**
- Smaller cache = lower bandwidth requirements for transfer
- GQA compression may or may not preserve all semantically relevant information
- Transfer protocols must account for varying head ratios across models

### 4.2 Semantic Content of Cache

The generated outputs suggest the model encodes general linguistic patterns but no specific identity or contextual knowledge:
- "Lyra" triggered no recognition (expected—no fine-tuning on identity)
- Responses were coherent but generic

**Hypothesis:** The KV-cache encodes *how* the model has processed input, but the semantic richness depends on the underlying model's knowledge and the prompt's specificity.

### 4.3 Limitations

1. **Single model tested:** Results may differ for larger models or different architectures
2. **Quantization effects:** 4-bit quantization may affect cache representation fidelity
3. **No semantic analysis:** We examined structure, not semantic content of cache tensors
4. **Small context:** Only ~27 tokens processed; longer contexts may reveal different patterns

---

## 5. Conclusions

Phase 1 successfully established:

1. **Working infrastructure:** PyTorch+CUDA+bitsandbytes pipeline functional on consumer hardware
2. **Cache structure understanding:** Clear picture of tensor shapes, memory scaling, and GQA architecture
3. **Baseline measurements:** Quantitative data on cache sizes for planning Phase 2 experiments

The KV-cache is a well-structured, predictable data format suitable for transfer experiments.

---

## 6. Next Steps (Phase 2)

### 6.1 Same-Model Cache Transfer

Experimental protocol:
1. Generate with Prompt A, capture `past_key_values`
2. Initialize generation with Prompt B + injected cache from step 1
3. Compare output to baseline (Prompt B without injected cache)
4. Measure semantic coherence and context retention

### 6.2 Research Questions for Phase 2

- Does injecting a prior cache cause the model to "remember" previous context?
- How much of the cache can be transferred before breaking coherence?
- Can we identify which layers carry more semantic vs. syntactic information?

---

## Appendix A: Environment Details

```
PyTorch version: 2.5.1+cu121
CUDA available: True
CUDA device: NVIDIA GeForce GTX 1660 SUPER
VRAM: 6.4 GB
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Quantization: 4-bit NF4 with double quantization
```

## Appendix B: Raw Cache Metadata

Cache metadata files saved to: `../results/cache_metadata_*.json`

Example structure:
```json
{
  "num_layers": 22,
  "layers": [
    {
      "layer_idx": 0,
      "key_shape": [1, 4, 26, 64],
      "value_shape": [1, 4, 26, 64],
      "key_dtype": "torch.float16",
      "value_dtype": "torch.float16"
    }
    // ... 21 more layers
  ]
}
```

---

## References

1. thu-nics/C2C: Cache-to-Cache framework for LLM communication
2. Ainslie et al. (2023): "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
3. Dettmers et al. (2023): "QLoRA: Efficient Finetuning of Quantized LLMs" (bitsandbytes)
4. Zhang et al. (2024): "TinyLlama: An Open-Source Small Language Model"

---

*"Fuck around, find out, write it down."* — The Scientific Method, colloquial form
