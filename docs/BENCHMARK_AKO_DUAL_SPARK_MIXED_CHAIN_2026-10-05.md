# AKO dual-Spark mixed attention and decode chains

## Scope and parallel execution

Both GB10/sm121 hosts were reachable and idle. Independent user-systemd jobs
started on both at 15:48:50 CST: .179 measured three ordinary/partitioned
decode cells, while .178 measured two complete mixed attention chains. .179
then independently repeated the mixed measurements; device-local tuning data
must not be copied from .178 into .179's selector.

Each probe uses MemoryMax=8 GiB, MemorySwapMax=0, RuntimeMaxSec=600 and a
32 GiB MemAvailable reserve. Every trial returned zero, released its CUDA
resources and observed an idle GPU. No production service was modified.

The finite budget was two mixed shapes and three decode shapes. Five paired
trials alternate order, each timing 30 complete launches/chains after warmup.
These are warmed-buffer attention measurements, not full-model serving times.

## Mixed chain measurement

The [preceding C2 trace](BENCHMARK_AKO_C2_BATCH_MARGINS_2026-10-05.md)
identified a final mixed step with **1,536 prefill queries plus one decode
query**, prefill history 30,976 and decode history 32,512. Its fallback selected
split-prefill owner 69 and executed mixed companion owner 70, because the
measured table contained pure-prefill records but no mixed records.

The new `--mixed-chain DECODE_DIRECTORY --decode-history N` probe reproduces
that unequal-history shape. It excludes decode rows from prefill tile metadata,
retains the original CSR offsets for both merge and decode, and executes the
same ordinary c468 decode companion after both prefill alternatives. The second
shape (2,048 prefill queries plus one decode query) is an additional bucket
representative, not a claim that this exact step was observed in that trace.

| GPU / queries / histories | Unsplit prefill + decode, ms | Split partial + merge + decode, ms | Unsplit reduction |
| --- | ---: | ---: | ---: |
| .178 / 1536+1 / 30976,32512 | 7.048 | 21.545 | 67.3% |
| .179 / 1536+1 / 30976,32512 | 7.292 | 21.645 | 66.3% |
| .178 / 2048+1 / 28672,32512 | 9.101 | 22.370 | 59.3% |
| .179 / 2048+1 / 28672,32512 | 9.349 | 22.682 | 58.8% |

Values are medians of unprofiled CUDA-event timings for the **whole attention
chain**, not the partial kernel alone. The selected unsplit prefill is c322;
the selected split prefill uses two partitions. Both include the ordinary c468
decode symbol. The mixed 1537-token workload captures a 2048-token runtime
bucket: unsplit grid 63×16×1, split grid 32×16×2, decode grid 32×8×1.

All four mixed cells passed full differential output/KV checks and a sampled
independent FP64 oracle under the 0.003 BF16 tolerance. For the traced shape,
maximum differential error was 0.000488281 and oracle error 0.000373563.
Outputs were **not bitwise identical**; this is not greedy token-vector parity
or model-quality qualification. Synthetic varied BF16 operands represent the
traced shape, not a replay of that model's captured layer values.

The traced mixed chain also passed Compute Sanitizer memcheck, racecheck and
synccheck on .178: zero errors, hazards or warnings. This checks the new offline
launch composition, not every production shape. No new hardware counters were
collected; the preceding split-prefill resource/stall capture remains supporting
context rather than a new mixed-chain counter claim.

## Independent .179 decode confirmation

| Rows / prior history | Ordinary c468, µs | c468 partitioned partial + merge, µs | Reduction |
| --- | ---: | ---: | ---: |
| 1 / 32513 | 1353.352 | 569.337 | 57.9% |
| 2 / 32513 | 1354.019 | 1147.770 | 15.2% |
| 2 / 32515 | 1330.413 | 1158.897 | 12.9% |

All three cells were bitwise equal with maximum differential error zero,
passed the sampled oracle and left KV unchanged. These reproduce .178's
preceding 13–14% C2 kernel-time advantage on the actual serving machine.
They do not prove an equivalent percentage reduction in end-to-end time.

## Changes and next boundary

Only offline benchmark code changed:

- `selected_policy_probe.cu`: explicit unequal-history mixed-chain composition,
  prefill-only metadata, and complete-chain timing/correctness.
- `ako_dual_spark_chain.mbtx`: pure workload/contract construction, bounded
  sanitizer capture and terminal verification. No runtime dependencies added.

Warning-denied native script checks passed; the new contract test and three
existing paired-trial parser tests passed (4/4). Both CUDA probe builds passed.
The remote MoonBit dependency's C compiler emitted an existing unused-result
warning; no full clean-suite claim is made in this unrelated dirty checkout.

**No route table, runtime bundle or production kernel was changed.** Next:
bind .179's own complete mixed-chain observations through the existing startup
table, preserve prior pure-prefill records, verify actual mixed-owner dispatch,
and run matched uninstrumented serving A/B with per-request token vectors.
The concurrent BF16 trajectory question remains separate and unresolved.

## Preserved results

Remote roots:

- .179 decode: `/home/wlc004s/lunaflux-ako-chain-decode-20261005.VolO55nU`
- .179 mixed: `/home/wlc004s/lunaflux-ako-chain-mixed-20261005.gc89OHfU`
- .178 mixed: `/home/wlc003s/lunaflux-ako-chain-mixed-20261005.5nGQmiJ9`

All three archives were downloaded without overwrite into
`/tmp/lunaflux-dual-chain-20261005.RiYHkG0W`. Their local SHA-256 hashes match
the remote hashes; every extracted manifest entry verifies:

| Archive | SHA-256 |
| --- | --- |
| decode179.tar.gz | `7db2b2fa1036a1609e18dd7da6223dcd4ea78f8967011bb8295cca328f468ab5` |
| mixed179.tar.gz | `5b46496b33c0bb7cbecb95612d9a7c5b3d11edb4549e9aa581eba6be7e771f23` |
| mixed178.tar.gz | `6e82cc744ca73b1b529ce4a40a28db609079df9c05bcdb5bf3ae60ecc5bf57a8` |
