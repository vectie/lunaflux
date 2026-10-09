# Indexed sparse execution feature

The old source generators used `maximum_tokens` both for retained history and
query/output capacity. A decode invocation with one query and many keys could
not bind compact query and selection storage. The new optional
`maximum_queries` defaults to the old value, preserving existing callers, but
controls only query count, output clearing and selected-index row storage.
History positions, pool capacity, validity, K/V and causal filtering retain the
independent history bound.

The functional split is:

`model geometry -> IndexedSparseAttentionPrecision -> CUDA source -> prepared frame -> caller queue`

The precision plan is immutable and contains no CUDA, GLM, device handles or
scheduler state. The CUDA source composes existing numeric implementations;
the prepared owner alone contains startup allocations and functions. Query
submission never generates code, computes layout or performs file inspection.

Producer stages: pooling -> deterministic stable top-k with visible incomplete
tail -> selected-index GQA attention. Consumer stages: attention only, borrowing
the producer's device allocation. Both share the same attention argument binder.
An IndexShare consumer owns zero pool/index scratch and never closes borrowed
histories or the producer's selection. A caller closes the shared queue first,
then consumers, then the producer.

At GLM index geometry 32 heads / dimension 128, pool 4, top-k 2048 + visible tail,
one query and history capacity 32768, owned scratch is:

| Buffer | Bytes |
| --- | ---: |
| Pooled keys | 2,097,152 |
| Pool raw positions | 131,072 |
| Pool validity | 32,768 |
| Pool count | 4 |
| Query selected indices | 8,204 |
| Total | 2,269,200 |

Borrowed projected histories are not included. In particular this does not
claim that an entire long-context model fits in this budget.

Validation: 96 affected native tests; producer/consumer queue regression checks
32 steps / 128 launches, zero measured hot allocations/blocking waits and active
cancellation/repeated closure. GPU source was exported by
`benchmarks/gpu_pipeline/export_indexed_sparse.mbtx`, and compiled on .178 GB10
UUID GPU-9c3d3cf0-439a-5da2-67e9-20414255879f with CUDA sm_121, `--fmad=false`.
Every physical fixture process was capped at 2 GiB, no swap, 120 seconds; no
other compute process was present before execution. The fixture's 15 live keys,
pool 4, top-k 8, two query heads / one KV head and dimension 4 cover causal
selection, incomplete tails, GQA and independent 8-query versus 1-query buffers.
Maxabs is zero, prefill/decode agree bitwise, idle output is zero. Memcheck reports
zero leaked bytes/allocations; memory/race/sync tools report zero errors. The
three-stage tiny decode fixture took about 30 microseconds outside sanitizers;
this is not model serving throughput or a performance comparison.

Remote artifacts: `/tmp/lunaflux-indexed-sparse.Q4ULaBgS` on .178. Downloaded
sources, binary and numerical/sanitizer logs:
`/tmp/lunaflux-indexed-sparse-check-20261009`. No checkpoint banks were loaded.
The downloaded CUDA header matches the locally exported header SHA-256
`6239fbc597e504944f7de9530cf6275d8d42ada508c1128a777c429dbecc05d8`.
Ordinary native check and the 96-test suite pass; warning-denied checking of this
feature is still blocked by the existing `internal/nccl` implicit `Eq` method
promotion warning. This feature does not modify that unrelated migration.

Remaining feature work: GLM DSA hidden-to-Q/K/V/index projections and rotary,
model IndexShare layer scheduling, complete decoder
and worker execution. This commit does not make those unfinished paths serve.
The current serial schedules are numerical reference implementations and need
a subsequent throughput schedule; they must not be called efficient serving
kernels merely because their execution wiring is complete.

## Request-owned projected history

`IndexedSparseHistoryPrecision` computes BF16 index/gate/K/V planes and metadata
without device handles. `IndexedSparseHistoryCudaSource` lowers three ordered
effects: reserve/reset and assign absolute query positions; parallel bit-exact
copies; publish valid rows and retained length after every copy CTA completes.
`IndexedSparseHistory` owns allocations/functions; `prepare_retained` composes
the three effects plus the existing three attention stages into one caller
queue. No host cache mirror, second completion or request-path allocation is
introduced. Startup zeros only metadata with a bounded 64 KiB scratch buffer.
The owner exposes its device query-position region and separate effect launches
so a complete decoder can order reserve -> projections/rotary -> copy/publication
without rediscovering positions on the host.

The owner represents one contiguous request history, not a batch-local row
array. Query counts and projections are bound once; reset is an explicit flag
set for the first frame of a new request. Idle rows leave history unchanged.
The append descriptor is I32 `[base, live_rows, error]`; overflow produces error
and empty attention output instead of writing out of bounds or truncating.
Worker integration must propagate that device error before delivering output.
Separate owners isolate requests; shared queues/frames must close before cache
release. This is not paged/latent-cache compression or multi-request batching.
In particular expanded K/V storage is included in the budget, and cannot be
reported as merely the 2.27 MB index workspace of the earlier 32K example.

At dimension 4, one KV head, eight query rows and history 32, owned storage is
1,216 bytes; attention scratch is 580 bytes. The native regression composes six
launches and executes 32 steps / 192 launches without measured hot allocation or
blocking waits. It also checks independent owners and active cancellation.
The final affected release-mode native suite is 99/99; ordinary native check
passes, while the unrelated dependency warning migration remains open.

The GB10 fixture executes nonzero BF16 value/index/gate history across prefill
chunks and decode, compares every active output to the ordered BF16-probability
oracle, and checks prefill/decode bitwise equality. It also checks interleaved
independent requests, idle preservation, reset/reuse and exact-full/overflow
capacity. Memory/leak/race/sync tests report zero issues under the same
2 GiB/no-swap/120-second process limits. No model weights are loaded.
Artifacts are in `/tmp/lunaflux-indexed-sparse.puDHPTXq` on .178, downloaded to
`/tmp/lunaflux-indexed-sparse-history-check-20261009` (first fixture) and
`/tmp/lunaflux-indexed-sparse-history-final-20261009` (final fixture, including
direct chunked-prefill versus decode comparison). The generated header SHA-256 is
`98f3882597235e4b0930673e5d278343a7fd24da8e9ba7400b7f127529596b70`.
The final downloaded probe matches repository source SHA-256
`f2557ce3cf9829775dbcd10304719caa5bed66afcb026e87bcd8aae56e0fb4f0`.
