# Register epilogue and completion effects

This change removes unnecessary materialization rather than preserving the old
generated source. The direct sibling register epilogue has unique global-store
owners and no shared readers; its trailing warp fence was removed. The finite
ownership regression covers partial rows. The generated kernel passes eight
row extents (2/7/17/33/63/65/127/1024), sentinel checks, sampled scalar reference,
and memcheck/racecheck/synccheck in `/tmp/lunaflux-register.KMPF7z`.

The common fold effect plan now distinguishes completed prefetch from pending
asynchronous prefetch. Both issue before consumption; only pending transfers
need completion waits. Narrow synchronous copies no longer emit empty async
commit/wait groups. Ring-model tests cover all three modes and 2/3/4 slots.
No arithmetic reassociation or model-family condition is introduced.

## Down output ownership exchange

The CUDA lowering converts each independently owned accumulator pair to BF16,
exchanges the packed register values, and writes contiguous 32-bit output pairs.
It replaces shared-result stores, reads, address permutation and two warp
fences. The pointwise rounding and ordered dot products do not change. Dynamic
result scratch is no longer used and the companion launch now requests zero
dynamic shared memory; the resource planner accounts for that same contract.

RTX 5060 Ti, pinned CUDA 13.1.115, isolated MLP probe, five alternating timing
trials, median microseconds (not serving throughput):

| Tokens | Scope | Old | Register exchange + zero dynamic scratch |
| --- | --- | ---: | ---: |
| 127 | Down | 32.861 | 29.965 |
| 127 | MLP chain | 86.075 | 83.182 |
| 1024 | Down | 179.456 | 164.149 |
| 1024 | MLP chain | 534.966 | 519.694 |

The first exchange-only experiment retained 4096 dynamic bytes and measured
about 176.99 us for down at 1024 tokens. Removing the unused launch allocation
is therefore part of the optimization, not optional cleanup. These observations
do not by themselves identify the exact occupancy or instruction-stall cause.

Both experiments pass full-output bit comparison against the prior kernel and
sampled scalar checks at 32/127/1024 tokens. Memcheck, racecheck and synccheck
pass on the 127-token tail case. Results are in
`/tmp/lunaflux-mlp-exchange.5ZRDuc` and `/tmp/lunaflux-mlp-zero.pVYAKY`.
The probe uses the real projection lowerer but is not an end-to-end runtime
benchmark. Final integrated serving performance and the separate cross-batch
numerical diagnosis are not closed by this measurement.
