# Luna kernel profile priority

This startup/offline package admits digest-bound profiler captures against an
immutable model plan and exact device target. The original stateless path is
unchanged. A separate paged path admits only `PagedKeyValue` plans and uses a
dedicated shape contract rather than projecting cache-backed work into
`ExactExecutionShape`.

Paged admission consumes the already admitted `PagedKvLaunchContractSet` and a
selected profile ID. The model identity, device target, catalog scope/version,
exact existing profile capacities, complete `DeviceKvLayout`, KV geometry and
tokens per page are derived from that opaque authority; profiler input cannot
assemble replacements. Every observed operation must have an admitted launch
contract under the selected profile.

Paged shapes bind exact prefill/decode row counts, per-row query and completed
context lengths, first-query cache positions, packed page-table slices and
block counts. `LunaPagedPageTableTraceDigest` is only a caller-asserted digest
of raw offline profiler trace bytes. It does not authenticate the physical page
identities encoded by those bytes, prove page ownership, or grant execution
authority. The trace claim, derived layout/profile fields, workload, profiler,
model content and model plan are digest-bound into canonical capture bytes.
The selected launch identity additionally binds every covered content-addressed
entry point, launch dimension, ordered semantic operand, byte/alignment claim,
and workspace bound. Optional counter rows use a closed vocabulary, bounded
values and strict canonical ordering; like timing rows, they are observations
only and carry no profiler or runtime authority.
Rows are copied during construction and must be prefill-first with contiguous,
complete page-table slices. Observations must be sorted and unique by operation
and exact shape digest, remain within the paged kernel profile and model
ceilings, and exactly account for declared attributed self time.

Both admission paths identify the model operation with the greatest measured
self time; ties retain immutable model-plan order.

The capture is prioritization evidence only. It does not claim that a custom
kernel exists, that a microbenchmark wins, that end-to-end serving does not
regress, or that sanitizer, race, leak, and numerical gates pass. An offline
producer may bind its digest into later specialization evidence only after
those separate gates are present. LunaFlux never profiles or compiles in the
request path.

The two paths remain disjoint: stateless observations are rejected for paged
plans and paged observations are rejected for stateless plans. Neither path
collects profiles, compiles artifacts, authorizes kernel promotion, opens a
device, or establishes physical correctness, readiness, or serving evidence.
