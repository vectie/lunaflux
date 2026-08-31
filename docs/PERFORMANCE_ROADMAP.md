# LunaFlux performance roadmap

This roadmap records the optimization work that changes inference speed. It is
deliberately separate from release qualification and physical-evidence history.
The normal loop for this work is focused correctness tests plus a short Qwen
benchmark; broad sanitizer, soak, and release campaigns run only when their
boundary changes.

## Current measured position

On the available `sm120` host, dense Qwen3-0.6B BF16 with one request and the
same token-ID input measured approximately:

| Runtime | Two-token latency | Relative to LunaFlux |
| --- | ---: | ---: |
| vLLM | 11.63 ms | 0.46x |
| SGLang | 21.60 ms | 0.85x |
| LunaFlux | 25.46 ms | 1.00x |

This is a small-model, short-prompt, concurrency-one measurement. It is useful
for decode launch and kernel iteration, but it is not a throughput conclusion.
Long prefill, long decode, shared-prefix, and concurrent Qwen workloads remain
separate profiles.

The result is nevertheless informative: the MoonBit scheduler, worker
protocol, paged KV owner, and whole-step execution graph are already close to
SGLang before deep numeric optimization. The remaining gap is primarily in the
GPU program:

- a full Qwen3-0.6B step still expands to roughly 311 launches because most
  semantic operations remain standalone;
- projection uses one hand-written shape policy rather than an autotuned
  backend portfolio;
- mixed paged attention assigns one warp to a query head and serially walks
  key positions instead of using distinct tiled prefill and split decode
  kernels;
- production fusion contracts exist, but the benchmarked Qwen deployment does
  not yet replace the corresponding graph spans;
- one maximum-capacity execution graph relies on count guards instead of a
  small set of batch/shape buckets.

## Portability boundary

Optimization decisions are tile-native and backend-neutral:

~~~text
model plan + live shape class
  -> LunaTile operation and fusion graph
  -> shape/capability strategy selection
  -> backend lowering
  -> AOT kernel and execution-graph bucket
~~~

The strategy vocabulary must not name Qwen, CUDA, warps, WMMA, or a vendor
library. A backend lowering may map the selected strategy to CUDA/CUTLASS,
cuBLASLt, HIP, or another device implementation. CUDA Graph is therefore one
backend realization of the generic execution-graph bucket contract, not the
public architecture.

## Optimization order

### P0: shape-aware projection

1. Make decode GEMV, small-row tiled GEMM, and large-row tiled GEMM explicit
   strategy classes selected from operation shape and live row bucket.
2. Add offline autotune records for the declared buckets (`1`, `2-8`, `9-16`,
   `17-32`, then larger prefill buckets).
3. Allow a backend portfolio: LunaTile generated kernels plus mature AOT
   library kernels where they win. GEMM is not reimplemented merely to keep it
   first-party.
4. Keep the token step allocation-free: selection resolves during startup and
   a bucket ID indexes prebuilt launches.

### P0: phase-specific paged attention

1. Decode uses a page-aware tiled or split-K strategy with vectorized KV loads.
2. Prefill uses causal tiled attention with online softmax and paged KV write.
3. Mixed batches compose the two prepared launches; neither phase pays the
   other's branch and launch geometry.
4. Prefix-reused pages and request-private tails retain the existing KV
   ownership contract.

### P0: production graph fusion

Wire the already typed patterns into dense Qwen plans when adjacency, layout,
and numerical contracts match:

- residual add + RMSNorm;
- Q/K normalization + RoPE;
- QKV projection + Q/K normalization + RoPE + positioned KV write;
- other fusions only after a step profile shows material benefit.

The plan builder recognizes semantic patterns. The selected backend decides
whether a fused implementation exists; model-family branching never enters the
executor.

### P1: execution-graph buckets and overlap

Prepare separate execution graphs for stable row/token buckets instead of
capturing only the maximum envelope. Continue overlapping CPU construction of
step `N+1` with device execution of step `N`, then remove avoidable host output
serialization where correctness permits.

### P1/P2: sampling, quantization, and speculative decoding

- keep greedy, top-k, and top-p selection on device;
- deploy measured FP8/I8/INT4 backends through the same shape selector;
- add draft/MTP verification as a scheduler capability, with tile-native
  verification kernels;
- add grouped GEMM and expert batching before using Qwen MoE as a headline
  benchmark.

## Benchmark loop

Every hot-path change first runs a short deterministic Qwen test with identical
weights, tokens, output length, sampling, GPU, and concurrency. Promising
changes then run:

- decode-heavy concurrency `1/8/16/32`;
- long prefill at fixed output length;
- long decode at fixed prompt length;
- shared-prefix and uncached controls;
- dense Qwen and, after grouped expert execution exists, Qwen MoE.

Report TTFT, inter-token latency, request throughput, output-token throughput,
and peak device memory. A microkernel win is kept only when it improves at
least one declared end-to-end profile without an unexplained material
regression elsewhere.
