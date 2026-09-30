# Offline projection observations

This package parses immutable tuning inputs; it performs no device queries,
filesystem operations or runtime profiling. Selection happens during AOT export.

## Fold pipeline records

`parse_projection_fold_records` accepts the following tab-separated format:

```text
luna-projection-fold-v1
scope<TAB>DEVICE<TAB>TOOLCHAIN_SHA256
record<TAB>BASE_SOURCE_SHA256<TAB>ARTIFACT_SOURCE_SHA256<TAB>MEDIAN_NS<TAB>SAMPLES<TAB>CHOICES
```

The caller provides the exact scope, including the device and complete compiler
policy/flags used for measurement. A source hash alone does not distinguish
different register ceilings or external compiler options. Each source family needs a baseline record
whose two hashes match and whose choices are `-`. An alternative's choices are
comma-separated `ROLE:STAGES:TRANSFER_TILES:FRAGMENT_STAGES`; roles are `matrix`,
`sibling`, or `intermediate`, with no repeated role. At least three samples and
positive median time are required. Compare matching workload vectors; a full
MLP-chain measurement must not be mixed with gate-only or down-only timing.

The CUDA AOT consumer regenerates a candidate before selecting it and compares
its source identity. Unsupported shared-memory combinations are excluded.
Unmeasured source families retain the baseline. Equal-cost baseline wins;
equal-cost alternatives have deterministic source-identity ordering.
The consumer defaults to a 1% minimum improvement over the baseline to avoid
switching schedules for small timing fluctuations. This explicit selection
parameter is independent of model family and does not alter raw observations.

`luna-projection-fold-v2` retains the same source binding and sample rules, and
encodes each choice as
`ROLE:STAGES:TRANSFER_TILES:FRAGMENT_STAGES:ROW_TILES:OUTPUT_TILES:HOIST`.
An optional eighth field selects finite sibling consumer groups:
`sibling:STAGES:TRANSFER_TILES:FRAGMENT_STAGES:ROW_TILES:OUTPUT_TILES:HOIST:GROUPS`.
Legal group counts are 4, 8, 16 or 32, bounded by the admitted strategy and
large enough to own every independent output tile. Group count is part of the
immutable schedule and source identity; launch threads are derived by the CUDA
lowerer, never overridden by a diagnostic launcher. Missing fields retain the
existing topology. The materialized down fold retains its separate geometry.
Zero row/output tile multiplicity retains the ordinary geometry; `HOIST` is
zero or one. Geometry multiplies independent row and output maps only, never
the ordered reduction. Address factoring retains the same canonical producer
coordinates while moving workgroup-invariant pointer and shared-offset
arithmetic outside the stage loop. Operand reads and publication remain at
their explicit effect boundaries. V1 remains readable unchanged; it cannot
encode or select these new executable alternatives.

V2 also accepts one `epilogue:retained` or `epilogue:coowned` token, alone or
alongside fold choices. This is a backend-neutral ownership/materialization
choice: retain the first fold and import its peer, or retain both values with
one consumer. It does not change reduction order or the pointwise numerical
contract. A missing token retains the measured split-plane baseline. Duplicate
or unknown ownership tokens are rejected. Ownership-only records are
alternatives, not baseline observations, and their entire generated module is
regenerated and source-bound before selection.

One optional `transport:segmented` or `transport:independent` token selects the
producer publication plan independently of epilogue ownership. Segmented
transport uses typed concatenated operand ownership and bounded load/publish
batches; independent transport retains complete worker traversal per operand.
It does not change accumulator geometry, reduction order, or SiLU rounding.
The missing token preserves existing executable defaults. Duplicate/unknown
tokens and v1 transport encodings are rejected, and transport-only records
are alternatives requiring a matched baseline. The device lowering consumes
the resolved plan for both transfer code and lifetime effects. Ordinary and
bounded row variants inherit the same policy; source binding covers both.
Neither alternative is preferred without complete-chain measured improvement.

The record does not imply that deeper pipelines are faster. Hardware timings
remain authoritative, and this format does not replace query/history/batch
attention routing or a complete serving benchmark.

`luna-projection-fold-v3` additionally accepts
`ROLE:STAGES:TRANSFER_TILES:FRAGMENT_STAGES:ROW_TILES:OUTPUT_TILES:HOIST:CONSUMER_GROUPS:MAX_TOKEN_ROWS`.
Zero row/output values retain the default extent. The last one or two fields
may be omitted: omission retains default cooperation or an unbounded domain,
respectively; explicitly supplied group counts and bounds must be positive.
A global choice and bounded choices may coexist for each role. Duplicate role/domain
pairs are rejected. The pure planner selects the narrowest applicable domain,
independently of record order, before lowering each AOT row variant. Thus a
small-row product does not silently replace a separately measured large-row
schedule. Source binding covers the entire module, including all variants and
the companion fold. V1 and V2 remain readable with their original semantics.

## Resource-policy observations

`parse_projection_resource_records` reads the offline
`luna-projection-resources-v1` table. The second line is
`scope<TAB>device<TAB>toolchain<TAB>shape<TAB>program`; the third is
`budget<TAB>max_threads<TAB>max_shared_bytes<TAB>max_registers_per_thread<TAB>min_resident_groups`.
Each following line is
`record<TAB>policy_id<TAB>source_sha256<TAB>latency_ns<TAB>samples<TAB>registers_per_thread<TAB>resident_groups`.
Tables end in a newline, require at least three timing samples, and reject
duplicate policy/source identities. Hardware observations must be collected
externally, not inferred from a policy identifier.

The Qwen candidate exporter accepts `--ingress-policy-search` to emit the bounded
backend-supported stage/window/fragment/accumulator frontier. Normal exports do
not build that frontier. `--ingress-policy-records PATH SHA256 DEVICE` enables
measured selection; it checks the saved scope, matches the baseline and each
alternative's freshly emitted source, and intersects supplied resource ceilings
with backend limits. Missing/stale baseline records or unknown constrained
resources cannot promote a candidate. No records preserves the labelled
unmeasured default. Export output prints the exact baseline scope, policy IDs,
source digests and candidate directories needed to construct a replayable table.

New `luna-projection-resources-v2` tables retain the same scope/budget lines,
then exactly one
`workload<TAB>phase<TAB>queries<TAB>rows<TAB>history<TAB>bucket_queries<TAB>bucket_rows`
and `provenance<TAB>physical_uuid`, before observations. The device scope is an
explicit performance compatibility class, while UUID is provenance only.
The parser and exporter derive a workload-specific pure selection scope.
Mixed workload aggregates are not accepted as v2 observations. Legacy v1 input
remains readable; it must not be relabelled as exact-bucket measurement.
