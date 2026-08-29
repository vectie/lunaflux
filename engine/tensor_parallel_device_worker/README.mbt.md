# Tensor-parallel rank-local device worker

This package is the explicit native owner for one authenticated local
tensor-parallel rank. It rebuilds the immutable physical execution plan from
root-derived model, sharded-file, launch-contract, device-plan, and artifact
evidence, then authenticates that exact digest through the decoded rank
bootstrap envelope. An envelope is never accepted as execution authority by
itself.

Preparation owns one exact non-optional live resource set after startup:

1. device context;
2. rank-local sharded weights streamed directly to their final allocation;
3. reusable rank-local plan descriptors;
4. encoded rank constants and activation/workspace storage;
5. one rank-local persistent KV arena;
6. stream, admitted AOT modules/functions, and prebuilt kernel arguments;
7. one ordered kernel executor with long-lived native leases and one event;
8. the generic tensor-collective communicator.

Cleanup is the strict reverse dependency order. Partial construction uses a
separate optional record solely so a pre-readiness failure can retain every
remaining cleanup authority. The warmed owner contains no optional resource
capabilities. A staged plan is retained only as scalar epoch, sequence, row,
token, operation, and collective cursors; the exact descriptor capability is
rederived and authenticated when needed.

Rank zero alone owns completion-frame storage, BF16 logits readback, and the
fixed sampling scratch required to publish the canonical worker completion.
Followers can return only an opaque generation/sequence/rank acknowledgement.
Kernel records and collectives enqueue on the exact same retained stream. No
operation performs per-kernel synchronization: after the final collective the
executor records one event, and only its nonblocking poll can publish executed
state. Fault cleanup synchronizes through the executor before releasing its
leases. Collective and rank-execution failures remain separately classified
for the rank-group wire protocol.

This is a software ownership and dispatch foundation. It does not claim that a
physical NCCL rendezvous, CUDA kernel bundle, multi-GPU numerical comparison,
sanitizer/leak probe, or performance gate has passed.

The phase boundary must run the package-native check/test matrix plus
`scripts/validate-tensor-parallel-device-worker-boundaries.sh`, the generated-C
warmed allocation scan, and
`scripts/validate-cuda-ordered-executor-sanitizer.sh`. Physical CUDA/NCCL
correctness, leak, soak, and benchmark evidence remains a separate gate.
