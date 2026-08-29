# Tensor-parallel AOT launch contracts

`kernels/tensor_parallel_launch_contract` admits one inert rank-local launch
contract for every exact semantic operation and bounded paged profile. It does
not reconstruct a model-family graph. Instead, it cross-authenticates the
immutable model operation sequence against the generic ordered operation-local
geometry in `model/tensor_parallel_plan`, including capability, input/output
and inner widths, local attention heads, head dimension, and collective site.
The current Llama builder is one producer of that generic evidence; another
dense-decoder builder needs no branch in this package.

Admission also binds the model identity, expected rank and world, device
target, catalog version, canonical equal-sharding v1 tensor regions, and exact
local `DeviceKvLayout`. Every supplied entry point must retain the resolved
catalog binding's content-addressed AOT family. Missing, extra, reordered, or
substituted operations, tensors, profiles, artifacts, or collective sites fail
startup. World size one is intentionally outside this rank-local launch path.

Callers repeat only exact operation-local launch geometry and bounded profile
scalars. Admission compares those values with the authenticated generic rank
plan. Operand order, roles, activation widths, local weight spans,
runtime-metadata spans, persistent K/V component spans, and maximum workspace
are derived internally. Sum all-reduce and last-axis all-gather are generic
physical primitives; a model-family builder decides which semantic operation
uses them. Sharded weights and K/V geometry must be strictly rank-local for
`world_size > 1`, with no full-replication fallback.

Every launch shares one `RankConstants(version=2)` operand after its runtime
inputs and before semantic graph inputs. The 32-byte, 16-byte-aligned record
contains eight little-endian Int32 cells: version, rank, world, compute major,
compute minor, K/V layout version, tokens per page, and one reserved zero.
Operation dimensions live only in their exact operation contract. A later
device owner uploads this rank/runtime-wide record once and reuses it.

Admitted contracts have no public output constructor and contain no module,
function, device address, allocation, stream, communicator, scheduler,
compiler, fallback, or JIT authority. Artifact admission and resource
acquisition remain separate explicit boundaries.
