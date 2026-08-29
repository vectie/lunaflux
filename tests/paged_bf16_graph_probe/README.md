# Synthetic paged-BF16 complete-graph probe

This evidence-only executable binds twelve immutable AOT launches into the real
`device` ordered executor with capture required. The fixed semantic order is:

1. embedding lookup;
2. attention RMSNorm;
3. Q/K/V projection;
4. positioned Q/K RoPE;
5. mixed prefill/decode paged attention and split K/V writes;
6. output projection and residual add;
7. MLP RMSNorm, gated MLP, and residual add;
8. final RMSNorm and LM head.

Every intermediate buffer begins with a finite poison value. After each graph
cycle, the independent MoonBit CPU referee reconstructs the fixture without
calling CUDA,
checks all fourteen readback boundaries including both KV arenas and all logits,
then performs deterministic greedy sampling from the final producing row. The
graph is reset and reused for a bounded number of cycles; every native owner is
closed explicitly in reverse dependency order.

The remote-ready command is:

```sh
scripts/probe-paged-bf16-graph-cuda.sh /absolute/path/to/nvcc-13.1 32
```

It compiles actual `sm_120` ELF CUBINs from the hash-pinned pointwise fixture,
the existing QKV/dense/MLP/LM-head fixtures, and the existing paged-attention
fixture. It builds the MoonBit executable separately, requires captured mode,
and rejects any runtime stderr. This is synthetic correctness and lifecycle
evidence, not a production fallback, model artifact, shape matrix, accuracy
campaign, leak soak, or performance promotion.

Local gates:

```sh
scripts/validate-paged-bf16-graph-probe.sh
scripts/validate-cuda-ordered-executor-sanitizer.sh
moon check --target native tests/paged_bf16_graph_probe --deny-warn
```
