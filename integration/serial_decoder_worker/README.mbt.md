# Prepared contiguous decoder worker

`SerialDecoderWorker` owns fixed input/output ports and asynchronous transfer
pools around a separately prepared decoder executor. At startup the caller:

1. Prepares bounded request ports.
2. Builds its model executor using the borrowed input and sampled-token ports.
3. Binds submit/poll/close effects and sparse append-error descriptors.

The frame's immutable input count determines sample and error port locations.
An optional `Positions` input carries absolute rotary/cache positions without
changing legacy port sizes or adding warmed allocations. Binding an absent
optional port is rejected before indexing the prepared resources.

The model-family adapter supplies those effects; generic worker execution has
no model-family branch. `begin` stages an existing wire frame. `progress`
nonblockingly retires input copies, the model queue, then sample/error copies.
When it returns true, `completion` returns the canonical wire value without
allocating a boxed optional polling result. The plan owner must remain unchanged
until completion. The completion view retains the existing wire epoch rules.

Close drains transfer leases before closing their descriptor-owning decoder,
then releases ports. This preserves cancellation and deterministic release.
Normal progress performs no blocking synchronization. The native transfer/device
double tests allocation, failure, cancellation and memory-budget behavior; it
does not execute CUDA math or establish serving throughput.

GLM decoder slices can bind here, but a complete checkpoint still needs text
prefix/suffix composition and, when exceeding one host's memory, cross-host
stage composition. A slice's completion is not automatically full-model output.

## Rank-local execution

`SerialDecoderRank` reuses the worker's prepared input/sample/error ports and DMA
logic, but separates metadata retirement, explicit queue submission, local
retirement and pipeline commit. Ingress/interior ranks do not read token output;
terminal ranks validate sampled IDs. No rank emits a model completion. Only a
coordinator commits the step and publishes the canonical response. A peer failure
poisons retained history. Whole-model worker APIs reject rank-owned ports.
