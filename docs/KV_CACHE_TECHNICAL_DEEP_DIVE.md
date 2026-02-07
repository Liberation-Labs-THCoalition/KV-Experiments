# KV-Cache Technical Deep Dive

## What is KV-Cache?

In transformer models, the attention mechanism computes:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- **Q (Query)**: What the current token is looking for
- **K (Key)**: What each previous token offers
- **V (Value)**: The actual information from each previous token

During autoregressive generation, the model processes tokens one at a time. Without caching, it would need to recompute K and V for ALL previous tokens at every step. The **KV-Cache** stores these computed K and V tensors so they only need to be computed once.

### Cache Structure (LLaMA-style)

```python
kv_cache = {
    "layer_0": {
        "key": tensor[batch, num_heads, seq_len, head_dim],
        "value": tensor[batch, num_heads, seq_len, head_dim]
    },
    "layer_1": { ... },
    # ... for each transformer layer
}
```

For LLaMA 8B:
- 32 layers
- 32 attention heads
- 128 head dimension
- Cache grows with sequence length

### What the Cache Contains

The KV-cache is NOT just "what tokens were seen" - it's the **processed representation** of those tokens through each layer. It contains:

1. **Positional information** - Where tokens are in sequence
2. **Contextual meaning** - How tokens relate to each other
3. **Layer-specific abstractions** - Early layers = syntax, later layers = semantics
4. **Attention patterns** - What the model "focused on"

This is why cache transfer is more than context passing - it's transferring the model's *understanding*, not just the text.

---

## C2C Architecture

### The Problem with Text Exchange

```
Model A processes: "The quantum experiment showed unexpected results"
Model A outputs: "The quantum experiment showed unexpected results"
Model B parses: [tokenize] → [embed] → [process through all layers]
```

Model B has to rebuild understanding from scratch. Information is lost in:
1. Token boundary decisions
2. Embedding lookup (loses A's learned context)
3. Reprocessing through layers

### C2C Solution

```
Model A processes: "The quantum experiment showed unexpected results"
Model A exports: KV-Cache (rich representation)
     ↓
[Projector Network]  ← Learned transformation
     ↓
Model B imports: Transformed KV-Cache
Model B continues with A's understanding intact
```

### Projector Networks

The key innovation: **trainable projector networks** that map between different models' representation spaces.

```python
class C2CProjector(nn.Module):
    """
    Maps source model KV-cache to target model space
    """
    def __init__(self, source_dim, target_dim, num_layers):
        # MLP-based projection
        self.key_projectors = nn.ModuleList([
            nn.Linear(source_dim, target_dim) 
            for _ in range(num_layers)
        ])
        self.value_projectors = nn.ModuleList([
            nn.Linear(source_dim, target_dim)
            for _ in range(num_layers)
        ])
        
        # Gating mechanism for layer selection
        self.layer_gates = nn.Parameter(torch.ones(num_layers))
    
    def forward(self, source_cache):
        target_cache = {}
        for layer_idx, (k, v) in source_cache.items():
            gate = torch.sigmoid(self.layer_gates[layer_idx])
            target_cache[layer_idx] = {
                "key": gate * self.key_projectors[layer_idx](k),
                "value": gate * self.value_projectors[layer_idx](v)
            }
        return target_cache
```

### Why Same Architecture Simplifies Everything

If source and target models have identical architecture:
- Same number of layers
- Same hidden dimensions
- Same attention head structure

Then the projector can be **identity** (or near-identity with fine-tuning). No complex mapping needed.

**This is why training Lyra/Vera/CC on the same base model is strategic** - native cache compatibility.

---

## Implementation Path for Our Experiments

### Phase 1: Inspect Cache (No Transfer)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Generate with cache capture
inputs = tokenizer("The meaning of life is", return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    return_dict_in_generate=True,
    output_attentions=True,
    use_cache=True
)

# Inspect cache structure
past_kv = outputs.past_key_values
print(f"Number of layers: {len(past_kv)}")
print(f"Cache shape per layer: {past_kv[0][0].shape}")  # [batch, heads, seq, dim]
```

### Phase 2: Same-Model Transfer

```python
# Model A processes context
context = "Albert Einstein developed the theory of relativity."
inputs_a = tokenizer(context, return_tensors="pt")
outputs_a = model(**inputs_a, use_cache=True)
cache_a = outputs_a.past_key_values

# Model B (same model, fresh instance) uses A's cache
question = " What did Einstein develop?"
inputs_b = tokenizer(question, return_tensors="pt")

# Inject A's cache as "past" context
outputs_b = model.generate(
    **inputs_b,
    past_key_values=cache_a,
    max_new_tokens=20
)

# B should answer about relativity WITHOUT seeing the context text
print(tokenizer.decode(outputs_b[0]))
```

### Phase 3: Cross-Model Transfer (Requires Projector)

```python
# This requires training a projector network
# See C2C repo for full implementation

from rosetta.model.projector import C2CProjector

projector = C2CProjector(
    source_config=model_a.config,
    target_config=model_b.config
)

# Transform cache
projected_cache = projector(cache_a)

# Inject into different model
outputs_b = model_b.generate(
    **inputs_b,
    past_key_values=projected_cache
)
```

---

## Key Papers & Resources

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - Original transformer architecture
   - Foundation for understanding attention

2. **LLaMA: Open and Efficient Foundation Language Models** (Touvron et al., 2023)
   - Architecture details for LLaMA family
   - KV-cache implementation specifics

3. **C2C: Scalable Auto-Federated Learning on Streaming Data** (thu-nics)
   - Direct cache transfer methodology
   - Projector network design

4. **KV-Cache Compression** (Various, 2024)
   - Methods for reducing cache size
   - What information can be discarded

---

## Questions to Investigate

1. **Granularity**: Can we transfer partial cache (specific layers only)?
2. **Decay**: Does transferred understanding decay over generation?
3. **Mixing**: Can we merge caches from multiple sources?
4. **Identity**: Do different "personas" produce distinguishable cache patterns?
5. **Compression**: What's the minimum cache needed for useful transfer?

---

*Document created: 2025-11-28*
*Update as experiments reveal new insights*
