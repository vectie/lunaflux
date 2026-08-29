# Tensor-parallel worker bootstrap

This startup-only package binds one complete same-host tensor-parallel group to
the exact architecture-neutral semantic model plan, nonzero model and group generations,
admitted topology, rank-specific worker startup contract, authenticated local
weight/device plan, physical execution plan, collective contract, and
rank-local KV binding. It accepts
only worlds 2 through 16; the existing single-worker bootstrap remains the
world-one oracle path and does not create a collective group.

The KV input is the output-only `TensorParallelKvRankBinding` produced by
`engine/tensor_parallel_kv_plan`. Each rank therefore binds a different local
KV arena and reservation while every binding retains the same logical PageId
geometry. This package never accepts or reconstructs a canonical full-head KV
layout per rank.

Admission produces one immutable rank-ordered group identity and one envelope
per rank. Device-plan, execution-plan, collective, worker, and KV digests are
rank-specific; the group digest is canonical over the complete ordered rank
set and identical in every envelope. The execution-plan package remains the
sole owner of its typed digest. This bootstrap carries only its canonical
SHA-256 wire form; the execution-plan v1 digest already binds the exact launch
contracts, rank constants, module digests, entry points, function symbols,
artifact geometry, physical operands, and collective sequence. No parallel
launch or artifact digest vocabulary is introduced here. Duplicate ranks,
incomplete sets, generation drift, ordinal or target substitution, and any
model/file/device/execution/collective/KV mismatch fail before an envelope is
published.

The version-two wire envelope is exactly 752 bytes. It carries only bounded
scalars, lowercase SHA-256 identities, and the opaque 128-byte NCCL rendezvous
identity. Caller-owned fixed frame buffers provide deterministic encode/load,
checksum validation, epoch-stale rejection, and bounded copies. Decoding does
not grant readiness or communicator authority; `authenticate_local` must match
the decoded envelope to current rank-local evidence before the private device
owner can consume its authority-free collective contract.

Physical `PlanBufferIdentity` values remain local to `RankGroupOwner`. They are
not serializable capabilities, so the envelope binds their exact model/group
generation domain and worker bootstrap identity without exposing an owner
pointer. The package exposes no launch set, artifact bundle, roots, file
locators, allocations, contexts, streams, communicator handles, scheduler
mutation, or descriptor-v1 changes. Digest admission is not device readiness
or physical execution evidence.

All collections and hashing are startup-only. The resulting group, envelopes,
digests, and rank-local contracts are immutable; no token-step or collective
launch path allocates through this package.
