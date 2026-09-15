# Grouped decode operand lifetime compaction

This experiment is rejected as a default optimization. Its scoped source and
results are preserved outside the production tree. It does not enable the
rejected broad split policy or alter the numerical fold.

The grouped asynchronous schedule already orders QK before its value-readiness
wait and workgroup publication. At that publication all K readers have retired.
The next K can therefore reuse one physical K tile while the current V remains
live. V retains two versions. No new publication or wait is introduced.

| Boundary | K | V |
| --- | --- | --- |
| Acquire key | Current tile published | Current transfer may remain pending |
| QK / acquire value | QK finishes; existing publication retires K readers | Current tile published |
| Prefetch next | Reuse retired K storage | Write alternate V slot |
| PV / release readers | Next transfer may remain pending | Current V readers retire |
| Terminal partial merge | Operand transfers drained | Reuse operand arena for partials |

`grouped_attention_operand_stages` is a pure device-neutral storage decision:
cooperative staging uses `(1, 1)` and the supported score-before-value pipeline
uses `(1, 2)`. The schedule exposes the same decision. CUDA lowering realizes
the offsets; it does not change model semantics, scheduling, or request behavior.

Resource accounting also follows actual ownership. Query, scores and running
fold values are register-owned, not additional shared allocations. Terminal
partial reduction reuses the retired operand arena, so the required capacity is
the maximum of operand-plus-validity storage and subgroup partial storage.

For head width 128 and eight subgroups:

| Schedule | Prior shared envelope | New shared envelope |
| --- | ---: | ---: |
| Cooperative, 64 keys | 34076 bytes | 32772 bytes |
| Async, 64 keys | 66844 bytes | 49160 bytes |
| Async, 32 keys | 33820 bytes | 24584 bytes |

The finite asynchronous model covers zero through nine tiles, every invalid
tile position, and rejected missing waits, publications, value retirement and
key retirement. Source tests check the shared K address, alternating V slots,
and resource boundaries. Source identity snapshots intentionally change: they
test reproducibility of the new lowering, not equivalence to old source bytes.

Before promotion, compare deterministic outputs, sanitizer results and timing
against the existing decode kernel, then remeasure the full serving matrix.
The previous cross-batch reference-accuracy question remains separate.

## Physical results: do not promote yet

The isolated source tree `/tmp/lunaflux-key-storage.kztWEU` passed all 3,109
native tests and seven release builds. Fresh AOT export, compilation and runtime
materialization passed. `/tmp/lunaflux-key-storage-probe.f585oq` compared fixed
decode inputs with the integrated `3b2f25d4` runtime: all output bytes and KV
contents matched. The existing independent scalar referee also passed;
memcheck, racecheck and synccheck passed for the four-row, 511-history case.
This does not close the separate real-model cross-batch accuracy investigation.

Five alternating CUDA-event trials (three warmups, thirty launches per trial)
show a regression, despite unchanged 71-register allocation:

| Rows / initial history | Prior median | Compact K median |
| --- | ---: | ---: |
| 1 / 0 | 4.111 us | 4.145 us |
| 4 / 511 | 36.949 us | 38.865 us |
| 8 / 4095 | 352.369 us | 354.437 us |
| 16 / 4095 | 707.546 us | 862.039 us |

The full 12-cell serving matrix completed with one warmup and five measured
trials per cell. At 4096/64 C16 it fell from 231.57 to **222.71 output tok/s**
(4422 to 4598 ms); 512/64 C16 fell from 1395.10 to 1340.31 tok/s. All response
lengths passed. The known 3072-token final-token variation remains.

The same compact-K cubin was then measured with four dynamic shared envelopes.
No kernel instruction, input or arithmetic changed between these launches:

| Shared launch envelope | Reported blocks/SM | C16 / 4095 median |
| --- | ---: | ---: |
| 24584 bytes | 3 | 866.24 us |
| 32776 bytes | 3 | 862.64 us |
| 33820 bytes | 2 | 711.11 us |
| 49160 bytes | 2 | 710.99 us |

The paired prior kernel measured approximately 708 us. This isolates the large
regression to the launch-resource/residency change, not a necessity for extra
instructions or new synchronization. It does not identify every underlying
cache or memory-pipeline effect; no new instruction-counter capture was taken.
Do not infer that maximum occupancy minimizes completion time.

A diagnostic 64-key compact-K variant (82 registers, 49160 shared bytes, two
blocks/SM) measured 703.75 us versus paired 707.03 us at C16, but slightly lost
at C8 (353.71 versus 352.13 us). This is not enough to justify a universal
replacement. It retained the existing scalar-reference tolerance and passed;
the sampled outputs also happened to be bit-identical.

Smaller storage is a legal transformation, not sufficient proof of a better
schedule. The default implementation was restored. Future offline selection
must distinguish minimum live storage from a measured launch resource envelope
and compare complete shape buckets; do not hard-code the diagnostic 33820-byte
value as an optimization. The rejected implementation is recoverable from the
archive and the local `/tmp/lunaflux-rejected-key-storage-tracked.patch`.

Archive: `/tmp/lunaflux-key-storage-results-20260916.tar.gz`, SHA-256
`e44b275e8da7386404fffc6398756fff856b70941b40a3e7d457c77cfe2c1283`.
It contains the scoped compiler source, full test logs, serving results,
fixed-input/reference/sanitizer probes and envelope experiments; deployment
launch arguments and process metadata are excluded.
