# Joint decode selection and runtime binding

The measured winner now reaches the composed AOT module and the immutable startup
descriptor. It is opt-in, not a global serving change. Same-module controls show
why: two partitions help single-row long decode substantially, but lose at C8.
This is a complete attention-chain experiment, not an end-to-end token/s result.

## Exact experiment

- Spark .179, GB10 sm121; UUID `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`,
  PCI `0000000F:01:00.0`.
- Source base `b27f45ea` plus this binding increment; the archived overlay and
  executable/source digests identify the tested code independently of its commit.
- Qwen3-0.6B BF16 attention geometry: 16 query heads, 8 KV heads, head dimension
  128, page size 8. Inputs are deterministic synthetic values and physical pages
  are permuted. These are homogeneous decode rows, not ragged/mixed serving.
- CUDA 13.0.88: `-O3 -arch=sm_121 --fmad=false --ftz=false --prec-div=true
  --prec-sqrt=true --maxrregcount=255 -lineinfo`. The winning c468 realization
  uses KV32, two stages and owned8 non-contracting F32 probability arithmetic.
- Each comparison uses five alternating pairs, three warmups and thirty
  repetitions per timing. CUDA events cover the complete partial-plus-merge
  chain. These launches are not a captured production graph.
- Serialized GPU work, systemd user units limited to 8 GiB with no swap and a
  30-minute timeout; every trial requires 32 GiB MemAvailable reserve. Observed
  control-trial availability stayed above 115 GiB.

Tool hashes:

```text
nvcc               fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1
compute-sanitizer  5cba659e1cd603aa2370e103425f184368e6ed405da90ce85743c4399c324dbc
```

## Joint search

The real candidate exporter enumerated c466, c468, c480 and c482 against
workgroup targets 128, 256 and 512, yielding two, four and eight partitions.
The measured workload was C8 with 8,192 previous tokens, hence compiler context
8,193 including the current token. The twelve complete-chain medians were:

| Candidate | Two partitions µs | Four partitions µs | Eight partitions µs |
| --- | ---: | ---: | ---: |
| 466 | 1175.62 | 1168.57 | 1171.67 |
| 468 | 1166.11 | 1169.48 | 1169.91 |
| 480 | 1465.38 | 1539.54 | 1289.12 |
| 482 | 1189.60 | 1170.10 | 1180.43 |

All twelve passed the probe's numerical checks. c468/two partitions had the
lowest median and 532,480 bytes of scratch at the 32-row compiled envelope,
versus 1,064,960/four and 2,129,920/eight. The nearby SIMT timings overlap;
this does not establish that two partitions are universally faster than four.
The synchronous matrix alternative c480 was substantially slower in this sweep.

## Same module attribution

The first retest measured 1,159.64 µs for the composed split chain versus
1,233.91 µs for the frozen unsplit kernel, about 6% improvement. However, the
new module's **unsplit** entry also beat that frozen kernel by about 6.5%.
Source comparison found different score handling: the frozen source retained
scores/probabilities in registers and used shuffles; committed code materialized
them in shared memory with separate full-tile/tail loops. Candidate ID 468 and
the arithmetic-law name alone did not establish identical implementations.
Compiler flags matched. The old-baseline comparison therefore cannot isolate
the contribution of partitioning.

The decisive control uses the same CUBIN and compiler configuration for both
entries. Lower is better; percentage change is the median of paired changes.
History below excludes the current token.

| Workload | Unsplit median µs | Two-partition chain median µs | Paired time change |
| --- | ---: | ---: | ---: |
| C8, history 127 | 10.205 | 15.662 | 53.5% slower |
| C8, history 8,192 | 1154.63 | 1174.13 | 1.36% slower |
| C1, history 32,768 | 1319.83 | 651.61 | 50.1% faster |

Every paired C8/8K sample regressed by 1.1–2.8%. Every C1/32K pair improved
by 48.4–50.7%. The long single-row result is consistent with greater parallel
work, but no new counters were collected to apportion latency hiding, instruction
count or memory effects. It must not be called a 50% serving throughput gain.

Standalone-selected versus composed-selected output is bitwise equal; paired
timing changes lie within roughly ±1%, with median change 0.74% slower. The
ordinary entry also remains bitwise equal to the frozen reference on the tested
C8/8K fixture, despite different generated score handling.

## Numerical and safety results

The probe compares complete BF16 outputs, checks KV immutability and uses an
independent sampled FP64 oracle with a 0.003 maximum-absolute-error limit.
These checks are not a model-quality or exhaustive all-shape proof.

An initially stronger bitwise requirement stopped the short-context partition
control: difference 0.000244141, sampled oracle error 0.000242642. Its logs remain
preserved. Partition changes are alternative softmax reductions, explicitly
allowed in this offline search; they are not bitwise-preserving layout changes.
The continued controls retain the existing numerical bound for partition changes
and require bitwise equality for symbol-only adaptation and the ordinary-entry
identity control. No source or binary changed between these checks.

The composed selected chain passes deterministic repeated CUBIN compilation and
all four CUDA tools: memcheck with full leak check, racecheck, initcheck and
synccheck. Memcheck reports zero leaked bytes and zero errors. GPU processes are
absent after the campaign.

One earlier CPU-only build failed because a macOS tar overlay contained AppleDouble
`._*.mbt` metadata. A new overlay excluding extended metadata and a fresh source
directory fixed packaging; the failed directory and logs were retained.

## Architecture and validation

`--bind-decode-chain` requires explicit measured chain inputs. It preserves the
selected lowering/ABI/workspace while adapting stable runtime symbols. Binding
rejects incompatible ordinary/split arithmetic laws, KV tiles and pipeline stages
before publishing artifacts. Selection and composition are immutable operations;
there is no request-time tuning or new token-step validation.

The runtime builder now forwards all supported typed decode descriptors instead
of recognizing only two legacy laws. It no longer appends an incompatible older
split implementation to a module that already contains its own partial/merge
pair. Bundle v15 authenticates the exact partition count; startup prepares launch
grids and scratch from that count, replacing a hardcoded eight. Legacy bundles
keep their existing format and eight-partition interpretation.

The isolated snapshot passes `moon info`, formatting, native check and
3,472/3,472 native tests, using the existing warning exclusions
`-79-20-29-25-92-14`. The builder automation test and host geometry tests pass.
Regression coverage includes opt-in requirements, numerical incompatibility,
unchanged selected lowering, counts 2/4/8/64, invalid counts, legacy roundtrips,
CLI propagation and runtime launch/workspace agreement.

## Remaining serving work

Do not globally activate the partition winner from this C8 sweep. Add the
unsplit choice to the comparable execution domain, calibrate per-workload routes
against the exact composed module, and then repeat captured-graph and end-to-end
tests. Existing module-bound route measurements cannot be reused after replacing
the module. Broader row/history diversity and mixed steps remain necessary.
No new vLLM/SGLang comparison, production deployment or serving token/s claim is
made in this increment.

Remote evidence root:
`/home/wlc004s/lunaflux-joint-decode-20261008.mdBKwhrO`.
It preserves the sweep, selected module, timing records, commands, failed controls,
same-module retest, sanitizer output, source audit and validation logs. `FILES.sha256`
seals the exported evidence; the compressed archive is also verified after download.

Archive SHA-256:
`638e9e79e72054c705f4f212b5ccd0f76a9019c781d10bb817ac5115655765a8`.
Downloaded without overwrite to
`/tmp/lunaflux-joint-decode-evidence-20261008.2pjL2dbi/evidence-v1.tar.gz`;
the local SHA-256 matches the remote archive.
