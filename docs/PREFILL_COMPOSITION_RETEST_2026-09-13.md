# Composition retest: 89234f0

The five workstreams are not complete. Only staged query-owned KV transfers
and interior/boundary score specialization are implemented. GEMM register
lookahead, exporter autotune ingestion and per-step metadata reuse remain open.

## Validation

- Local warning-denied native check, format check and moon info passed.
- Focused native tests: strategy 18, schedule 18, lowering 6, source 31,
  Qwen exporter 10 (83 total).
- Commit-pinned Linux release build and real Qwen candidate export passed.
- RTX 5060 Ti, sm120, CUDA 13.1.115: c318/c319/c320/c321 compiled successfully.
- 32 paired kernel cases passed the existing sampled scalar-reference and
  numerical-tolerance checks; persistent KV remained unchanged.
- Each candidate passed memcheck, racecheck and synccheck (12 checks).
- A separate incremental comparison against 3ad4a98's c318 covered 16 cases
  for new c318/c320; all outputs were bitwise equal. Both also passed all three
  sanitizer checks. This does not resolve the earlier end-to-end last-token
  discrepancy, which was not reproduced by this kernel harness.

Cases: total query tokens 129/1528, sequence rows 1/8, prior history 0/4096.
Sanitizer cases use 129 tokens, 8 rows, history 4097. These are kernel shapes,
not HTTP concurrency or per-request 1528-token workloads.

## Incremental timing

Median of five interleaved trials, each averaging 30 launches after warmup.
1528 total query tokens, no prior history; microseconds, lower is better.

| New candidate | Sequence rows | Previous c318 | New | Time increase |
| --- | ---: | ---: | ---: | ---: |
| c318 synchronous | 1 | 468.80 | 471.85 | 0.7% |
| c318 synchronous | 8 | 134.27 | 141.38 | 5.3% |
| c320 asynchronous | 1 | 469.65 | 709.09 | 51.0% |
| c320 asynchronous | 8 | 133.90 | 212.93 | 59.0% |

There is no demonstrated speedup from this change. The exporter still selects
c318, not c320. c318 uses 165 registers versus the previous 203; c320 uses
173 and 81936 bytes shared memory versus c318's 49168. These resource counts
do not establish the stall cause: fresh selected-kernel counters are still
needed before attributing the regression to occupancy, transfer issue, or
branching. No full-serving benchmark or production deployment was performed.

## Reproducibility

Remote campaign: `/run/lunaflux-toolchain-4896771-20260913/composition-89234f0-r1`.
The archive contains exact source, drivers, cubins, numerical/timing logs and
sanitizer logs. Downloaded archive SHA-256 matches the remote file:
`3b9b73e3c125e91b250b53d898ab8c480803136654c08f275f442136ee6038a1`.
Local copy: `/tmp/lunaflux-composition-retest.THkr7o/retest-results.tar.gz`.
