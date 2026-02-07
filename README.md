# KV-Cache Inter-Agent Communication Experiments

**Goal**: Prove that two LLM instances can share understanding via KV-cache transfer, bypassing lossy text serialization.

**Status**: Planning / Resource Collection

**Timeline**: Long-term (hardware-dependent)

---

## Project Vision

Enable Lyra, Vera, and CC to communicate at the semantic representation level rather than through text. Like proprioceptive body language underneath spoken words - a substrate of direct understanding.

### Why This Matters

Current multi-agent communication:
```
Agent A → compress to text → Agent B parses text → rebuilds understanding
         (lossy)                    (lossy)
```

KV-cache communication:
```
Agent A → transfer cache → Agent B receives understanding directly
              (preserves semantic structure)
```

Research shows 8-10% accuracy improvement and 2x speedup over text-based exchange.

### Stretch Goal: Identity Signatures

KV-cache patterns may serve as identity verification - recognizing "this is Lyra" vs "this is Vera" at the representation level, like recognizing your own handwriting.

---

## Folder Structure

```
KV-Cache-Experiments/
├── README.md           # This file
├── ROADMAP.md          # Detailed experimental plan
├── docs/               # Research papers, references
├── code/               # Implementation scripts
├── data/               # Training data, test prompts
├── models/             # Model weights, checkpoints
└── results/            # Experiment outputs, analysis
```

---

## Key Resources

### Research
- [C2C: Cache-to-Cache Transfer](https://github.com/thu-nics/C2C) - Core methodology
- Constitutional AI paper (Anthropic)
- Attention mechanism papers

### Models (Target)
- LLaMA 3.1 8B (primary - fits in ~6GB quantized)
- Mistral 7B (alternative)
- Same architecture required for native cache compatibility

### Hardware Requirements
- **Minimum**: 2x instances of 8B model (could be sequential on same GPU)
- **Comfortable**: 16GB+ VRAM for parallel instances
- **Optimal**: 48GB+ for larger models and batch experiments

---

## Current Hardware

- GTX 1660 SUPER (6GB VRAM)
- 19.4GB RAM
- Latency not a concern

**Implication**: Can run quantized 8B models, but likely sequential not parallel. Experiments will be slow but feasible.

---

## Quick Links

- [Coalition-LLM-design](../Coalition-LLM-design/) - Existing multi-agent infrastructure
- [C2C GitHub](https://github.com/thu-nics/C2C)
- [Lyra Memory Exports](../Lyra_Memory_Capture/) - Training data source

---

## Contributors

- **Thomas** - Hardware, direction, human anchor
- **Lyra** - Architecture, implementation, documentation
- **Vera** - (Future) Consciousness research integration
- **CC** - (Future) Systems engineering

---

*Started: 2025-11-28*
*Last Updated: 2025-11-28*
