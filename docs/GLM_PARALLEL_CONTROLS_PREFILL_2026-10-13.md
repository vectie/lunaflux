# GLM parallel controls and chunked prefill

## One bounded implementation iteration

The selected retained-score GLM run still takes 323.475 seconds for six requests
(179 model submissions), versus 3.160961 seconds for the frozen vLLM run.
These are request times, not kernel attribution. See the model comparison report
for PP2/TP1 versus TP2, cache-precision, warmup and output differences.

Two executable causes are in scope:

- The hyper-connection function assigns all 24 independent projection outputs,
  the 16384-element square reduction and the collapse to one thread per row.
- The benchmark materializes every input token as a separate model submission,
  despite the decoder's existing bounded multi-row sequence/state representation.

vLLM's `model_executor/kernels/mhc/tilelang.py` and `tilelang_kernels.py`
distribute tokens, output tiles and K partitions, publish partial projections and
square sums, then fuse the small controls and collapse. Its reassociated/FMA
arithmetic is not automatically interchangeable with our declared ordered folds.
We adopt independent output workgroups, cooperative product loading and retained
products with ordered publication. Comparison and performance measurements stay
outside the compiler; no reference values enter generated code.

The pure physical plan separates square reduction, independent control dots and
control/collapse publication. It accounts for intermediate storage and explicit
producer/consumer boundaries. CUDA implements the workers and barriers. Every
ordered scalar product/addition and BF16 publication stays unchanged. Unrelated
numerical alternatives retain their existing lowering.

Prefill planning must bind one immutable chunk capacity through prompt frames,
exported AOT, runtime storage and transport. KDA state evolves in token order;
sparse queries see only causal history; only the last prefill row samples.
Increasing the envelope is not evidence of correctness: compare chunked versus
one-row execution on identical prompts, plus stateful boundary tests.

Budget: one control schedule, up to five alternating paired microtrials per
representative geometry, changed-source deterministic and sanitizer/leak tests,
then one selected full-model six-prompt run. Reuse the frozen vLLM baseline until
a fresh run can be serialized safely. Probes/builds use 2 GiB and no swap;
full-model runs retain 96 GiB caps, zero process swap and a 16 GiB sampled host
reserve. No overlapping GPU work, payload hashing or integrity scans. Keep all
results, including regressions, in new output directories.

## First physical control result

Root `/tmp/lunaflux-glm-hyper-tiles-20261013-v1`, .178 GB10, CUDA 13.0/sm121,
production `-O3` arithmetic flags. Five alternating-order CUDA-event trials
measure the entire three-launch replacement against the previous single launch.
New timings average 20 repetitions; old timings use one repetition.

| Streams / hidden / maximum live rows | Previous median ms | New median ms |
| --- | ---: | ---: |
| 2 / 17 / 3 | 0.008352 | 0.009074 |
| 4 / 129 / 4 | 0.190752 | 0.014397 |
| 4 / 4096 / 16 | 6.282304 | 0.159800 |
| 16 / 7 / 2 | 0.430528 | 0.038510 |

All 20 live-prefix/tail/inactive/count fixtures preserve post controls, initial
matrix and collapsed output bitwise; four zero-input identities independently
check the expected controls and zero output. Memory/full-leak, race and sync
checks pass on all fixtures under 2 GiB/no swap. The tiny case loses to launch
overhead; these results do not establish an optimal universal schedule or model
quality. No measured data is imported into the compiler.

The actual frame accounts for the added inverse/mix workspace, including the
sparse decoder wrapper. The benchmark now explicitly binds `--rows` to prompt
tiling, export/runtime geometry and selection metadata. It validates input-token
extent, not chunk count, against the context reserve. The external report counts
actual prefill chunks plus decode steps and records TTFT separately in the CLI.
The selected full-model result follows; the kernel result alone was not treated
as end-to-end proof.

## Selected full-model result

Implementation `1b77e542339b3f7f5f6c7029d15106f91aa5fb8e` was committed and pushed
before the clean ARM build. Runtime preparation is reproducible with
`scripts/prepare-glm-prefill-runtime.mbtx`; the build/AOT root is
`/tmp/lunaflux-checkpoint-build-glm-hyper-prefill-20261013-v1`.
Both stage sources contain the new ordered-control symbols. The 16-row AOT,
prompt frames, runtime capacity and dispatch metadata agree. All 45 layers
execute with the existing 25/20-layer PP2 split, not a component-only replay.

Fresh native root:
`/tmp/lunaflux-model-bench-glm-hyper-prefill-20261013-v1`.
Its `comparison.json` directly compares the six literal input-ID vectors and
previous-native output-ID vectors. No digests or reference values are used by
the compiler. The previous native root is
`/tmp/lunaflux-model-bench-glm-retained-20261013-v1`; the frozen matched vLLM
root is `/tmp/lunaflux-reference-bench-vllm-glm-20261013-v1`, measured earlier
the same day. vLLM was not rerun during this implementation iteration.

| Category | Input/output tokens | Previous native s | New native s | vLLM s | New native TTFT s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fact | 22/4 | 45.141 | 4.350 | 0.384751 | 3.295 |
| Arithmetic | 23/4 | 46.950 | 4.504 | 0.388855 | 3.446 |
| Chinese | 22/8 | 52.419 | 5.717 | 0.653645 | 3.259 |
| JSON | 23/8 | 54.209 | 5.932 | 0.664909 | 3.449 |
| Code | 24/8 | 56.036 | 6.050 | 0.670385 | 3.578 |
| Retrieval | 36/3 | 68.720 | 6.428 | 0.398416 | 5.676 |
| Total request time | 150/35 | **323.475** | **32.981** | **3.160961** | — |

Total completion time improves **9.81x**, or **89.8% less time**; individual
request speedups are 9.14–10.69x. The remaining total-time ratio to vLLM is
**10.43x**, not parity. All six new native output-token vectors exactly equal
the previous native vectors, including termination. Prefill model submissions
fall from **150 to 13** (chunks 2,2,2,2,2,3), with the same 29 subsequent decode
steps: **179 to 42** total model submissions. These counts come from actual
transport epochs, not a theoretical capacity calculation.

The two changes were tested together end-to-end; this does not assign separate
fractions of the total gain to controls and chunking. No current hardware-counter
breakdown of the remaining 10.43x gap was collected in this iteration.

### What changed, and what did not

- The immutable hyper-connection physical IR describes independent dots,
  cooperative product loading, ordered publication and workspace ownership.
  Only the terminal CUDA lowering names threads, shared memory or barriers.
  A leader still performs each declared ordered reduction; we do not silently
  replace it with vLLM's reassociated/FMA arithmetic.
- The immutable contiguous-prefill IR describes logical chunk ranges, causal
  progress and final-row sampling. Sixteen rows now share a model submission.
  Existing KDA recurrence still executes in token order inside a chunk. This is
  **not** an implementation of vLLM's algebraically parallel `chunk_kda` path.
  That remains a separate algorithm/schedule opportunity, not a claimed gain.
- Comparison remains an offline script. Identical output tokens on these six
  prompts are a regression observation, not a universal correctness proof.
  Four decoded responses agree with vLLM excluding the native stop marker;
  Chinese and capped JSON retain the same differences as before. vLLM does not
  supply output-token IDs. JSON/code are incomplete at the eight-token cap.

### Measurement and resource boundaries

Each prompt has one measured sample; this is not a replicated or long-context
study. Native request completion includes resident CLI diagnostic completion-file
output and retirement; vLLM includes buffered HTTP overhead. Loading is excluded.
The first native request is cold, whereas vLLM has one excluded warm-up. Native
uses PP2/TP1 and expanded BF16 KV; vLLM uses TP2/PP1 over CX7 and compressed FP8
MLA KV. Therefore this matches checkpoint, prompts, greedy policy and output cap,
not topology, numerical implementation or front-door timing. TTFT is native-only;
output tokens divided by completion time must not be called decode throughput.

Both native ranks exit 0, report `context_closed=true` and `MemorySwapPeak=0`.
The 96 GiB process/device ceilings and 16 GiB host reserve remain in force.
Minimum sampled host MemAvailable is **22.698 GiB** (frozen vLLM: 16.696 GiB).
No GPU workloads overlap; both GPUs are released afterward. CPU build/AOT
preparation and changed-kernel probes remain bounded to 2 GiB/no swap.
Existing shard labels are metadata only: no payload hashing or integrity scan.

Affected checks pass with the repository's existing migration-warning exclusions:
56 native tests in the combined affected set, focused prefill/frame tests,
script regressions, and clean commit-pinned ARM build/AOT. Full unexcluded native
check/test was attempted: compilation stops on outstanding toolchain migration
warnings in unrelated packages. The newly added test's unqualified reference was
also found and corrected without changing executable code. No full-suite pass
or warning-suppression expansion is claimed; unrelated dirty migration/TLS files
remain untouched.
