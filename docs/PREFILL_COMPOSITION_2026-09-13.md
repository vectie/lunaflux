# Prefill composition work

This work targets generic compiler schedules, not Qwen-specific shape branches.
It is incomplete and is not a new performance result.

| Workstream | Implementation status | Remaining validation/work |
| --- | --- | --- |
| Query-owned attention plus async KV | Added two-stage candidates 320/321 and disjoint K/V ring slots | CUDA compile, numerical tests, racecheck and timing; measured selection |
| Interior/boundary attention | Separate template instantiations of the ordered score fold | CUDA numerical and timing comparison; expf unchanged |
| GEMM register lookahead | Not implemented in this change | Immutable fragment lifetime plan, lowering, resource/timing tests |
| Offline autotune integration | Not implemented in this change | Exporter record ingestion and actual measured records |
| Per-step metadata reuse | Not implemented in this change | Shared descriptor lifecycle and runtime/kernel binding |

The async candidates require two stages and the complete shared-memory footprint.
They do not receive synthetic latency records. Existing stable candidate IDs are
unchanged. Source tests cover both dense-current and paged-history address maps.

The score specialization retains ordered maxima, expf, BF16 conversion and PV
accumulation. This is not a claim that the previously observed last-token
discrepancy has been resolved. That discrepancy still needs a logits-level check.

Local warning-denied native tests passed: attention strategy 18, tile schedule
18, CUDA lowering 6, CUDA source 31. These are compiler/source tests, not GPU
correctness or performance tests. No production deployment was performed.
