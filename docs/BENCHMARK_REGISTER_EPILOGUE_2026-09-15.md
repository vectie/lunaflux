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

## Integrated result: 3b2f25d4

The first `a73c7bb1` release binding failed because the execution layer still
required companion scratch to equal `block_x * 32`. Commit `3b2f25d4` fixes that
obsolete inference. Its clean native suite passes 3108/3108, all seven release
entries build, and release binding/materialization pass. The kernel tree is
unchanged from a73c7bb1, whose compiled modules were reused and rebound; the
runtime executables are freshly built from 3b2f25d4.

Uninstrumented Qwen3-0.6B BF16 serving, one warmup and five trials per cell:

| Input/output | C1 tok/s | C8 tok/s | C16 tok/s | Prior b9440271 C16 |
| --- | ---: | ---: | ---: | ---: |
| 512/64 | 211.921 | 1036.437 | 1395.095 | 1385.656 |
| 1528/32 | 149.533 | 373.178 | 412.571 | 409.600 |
| 3072/32 | 107.383 | 187.546 | 198.604 | 197.303 |
| 4096/64 | 117.216 | 217.502 | 231.569 | 230.527 |

4096/64/C16 improves only 0.45% in throughput. This does not reproduce the
isolated down improvement as a whole-service improvement. Actual selected
shapes and time contribution must be measured before attributing the dilution.
Historical vLLM/SGLang C16 values remain 266.38/260.71 tok/s (not rerun here),
so current completion time is still approximately 15.0%/12.6% higher.

All response lengths pass and the runner stops its owned service, leaving the
GPU idle. 3072/32 C8/C16 still show differences at token index 31 relative to
their first measured response; the other ten cells do not. This is not an
independent reference accuracy check, and it does not close numerical acceptance.

Remote source: `/tmp/lunaflux-integrated-3b2f25d4.XHVR7g`.
Downloaded E2E results:
`/tmp/lunaflux-a73c7bb1-download.U1HUO9/e2e-3b2f25d4-results.tar.gz`, SHA-256
`030d4d4fdf70b710baf85a18aab06849e2b93e6bc7e288de166ae0451aae0de8`.
Downloaded isolated experiments: the same directory's `register-results.tar.gz`,
SHA-256 `f53644b101f7a27f12e867f75d9040dacb8c08a4feabdd81b35ee65d36c4037c`.
Both hashes match the remote archives.
