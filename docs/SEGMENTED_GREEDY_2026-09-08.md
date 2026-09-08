# Segmented greedy reduction

The neutral sampling strategy partitions the vocabulary into disjoint segments.
Each segment produces an associative summary: maximum finite value, lowest
token index at that value, and lowest non-finite token index. Merging summaries
preserves the existing deterministic greedy contract without reassociating
floating-point arithmetic.

The CUDA backend lowers this plan to a partial reduction and a summary merge.
Both steps are prepared at startup and participate in the existing ordered
execution graph. For vocabulary 151,936 and a 32-row capacity, 38 segments of
4,096 values require 19,456 bytes of startup-owned scratch, in addition to the
unchanged 256-byte result buffer. This allocation uses the existing sampling
workspace lifecycle and must be included when accounting for runtime memory;
it does not change model activation-arena bytes. No token-step allocation is
introduced.

A version suffix on the admitted head entrypoint selects the two-step ABI.
Unmarked old head modules retain the original one-step reducer. Source exports
contain both versions, so no runtime symbol probing or opportunistic fallback
is needed.

## Isolated physical comparison

Exact emitted source was compiled and compared on the approved RTX 5060 Ti.
Numbers below are medians of three trials, with graph-replayed launches and
hot logits. New timings include both kernels; these are not end-to-end serving
results or a comparison to another framework.

| Active rows | Old reducer µs | Segmented reducer µs |
| --- | ---: | ---: |
| 1 | 70.112 | 5.671 |
| 2 | 70.270 | 5.777 |
| 8 | 70.196 | 8.606 |
| 32 | 70.378 | 21.266 |

The probe checks ties across segment boundaries and the vocabulary tail,
lowest-index non-finite rejection, invalid row ends, and untouched inactive
rows against an independent expected result for both implementations. Source,
probe and raw results are under
`/dev/shm/lunaflux-sampling-segments-20260908-r1` on the test host.

Memcheck passed. The initial racecheck including all timing graph replays
terminated unsuccessfully with no reported hazards; its logs are retained.
A separate correctness-only rerun passed racecheck with zero errors and
warnings. The exact kernel source was unchanged for that bounded rerun.
