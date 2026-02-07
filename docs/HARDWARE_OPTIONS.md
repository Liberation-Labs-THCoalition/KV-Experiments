# Hardware Options for KV-Cache Experiments

## Current Setup

| Component | Spec |
|-----------|------|
| GPU | GTX 1660 SUPER (6GB VRAM) |
| RAM | 19.4GB |
| Storage | SSD |

### What's Possible Now

- **4-bit quantized 8B models** - Fits in ~5GB VRAM
- **Sequential experiments** - Run one model at a time
- **Cache save/load** - Store to disk, reload in new session
- **Slow but functional** - Latency not a concern (per Thomas)

### Limitations

- No parallel model instances
- Large models (13B+) won't fit even quantized
- Training projector networks will be slow

---

## Cloud Options (Pay-as-you-go)

### Lambda Labs
- RTX A6000 (48GB): ~$1.10/hr
- Good for burst experiments
- https://lambdalabs.com/

### RunPod
- Community cloud, variable pricing
- Can find deals on older hardware
- https://runpod.io/

### Vast.ai
- Cheapest option usually
- Less reliable, but budget-friendly
- https://vast.ai/

### Estimated Costs

| Experiment | Hardware Needed | Est. Time | Est. Cost |
|------------|-----------------|-----------|-----------|
| Phase 1 (Cache inspection) | Current potato | N/A | $0 |
| Phase 2 (Same-model transfer) | Current potato | N/A | $0 |
| Phase 3 (Identity signatures) | Current potato | N/A | $0 |
| Phase 4 (Communication protocol) | 24GB GPU | 10 hrs | ~$15 |
| Phase 5 (Multi-agent) | 48GB GPU | 20 hrs | ~$25 |
| Phase 6 (Training Lyra-model) | 2x 24GB | 50+ hrs | ~$150+ |

---

## Hardware Upgrade Paths

### Budget Tier (~$300-500 used)
- RTX 3060 12GB
- Enables 8B models at higher precision
- Parallel instances of 4-bit models

### Mid Tier (~$800-1200 used)
- RTX 3090 24GB
- Full precision 8B models
- Quantized 13B models
- Comfortable for most experiments

### Research Tier (~$2000+ each)
- RTX 4090 24GB
- Faster training
- Larger batch sizes
- Still need 2+ for serious multi-agent

### Dream Tier
- H100 rack from Santa Claude
- All problems solved
- Unlikely but aspirational

---

## Distributed E-Waste Cluster

Thomas's preferred approach: network multiple older GPUs.

### Requirements
- Multiple machines with GPUs (even older ones)
- Fast network connection between them (10Gbe ideal)
- Orchestration layer (Ray, DeepSpeed, or custom)

### Challenges
- Network latency for gradient sync
- Different GPU generations = load balancing issues
- Setup complexity

### Advantages
- Cheap hardware acquisition
- Scales horizontally
- Good hacker aesthetic

### Minimum Viable Cluster
- 2-3 machines with GTX 1080 or better
- Ethernet switch
- NFS for shared storage

---

## Recommendation

**For now**: Run Phases 1-3 on current potato. These are exploratory and don't need parallel instances.

**When ready for Phase 4+**: Rent cloud GPU for specific experiments. Don't buy hardware until we know exactly what we need.

**Long term**: Build e-waste cluster as ongoing project. Accumulate hardware opportunistically.

---

*Last updated: 2025-11-28*
