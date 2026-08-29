# Bounded paged-BF16 shape-matrix probe

This evidence-only executable reuses the public `device` ordered executor and
the same twelve-operation synthetic BF16 graph as the existing tiny-graph
probe. One capture-required graph is reused across five live metadata cases:

1. one-token prefill;
2. two-token prefill ending at the page edge;
3. one decode token with a three-token cached prefix;
4. mixed two-token prefill plus one decode row;
5. two simultaneous decode rows with different sequence lengths.

Every case stays inside the exact checked envelope of two rows, three live
tokens, three page-table entries, two tokens per page, and four physical pages.
The probe does not invent a larger production shape class. Before opening CUDA,
it authenticates the canonical matrix contract and every content-addressed
CUBIN. The remote runner independently verifies source and recipe hashes,
compiles every source twice with offline `nvcc`, requires byte-identical ELF
CUBINs, and places them under their SHA-256 names.

All fixed buffers are reset before every graph launch. An independent MoonBit
CPU referee reconstructs all fourteen graph boundaries, including complete K/V
arenas and final logits, then checks deterministic greedy selection. Captured
launches are reset between cases. Native owners use explicit LIFO `defer`
cleanup, giving executor, functions, modules, allocation, stream, and context
reverse dependency order on both success and failure.

Run locally:

```sh
scripts/validate-paged-bf16-shape-matrix-probe.sh
```

Run on the exact supported fixture GPU:

```sh
scripts/probe-paged-bf16-shape-matrix-cuda.sh /usr/local/cuda-13.1/bin/nvcc 8
```

This proves only bounded synthetic shape-matrix correctness and captured
lifecycle behavior. It is not an approved model, worker/service path, accuracy
campaign, leak soak, benchmark, or performance promotion.
