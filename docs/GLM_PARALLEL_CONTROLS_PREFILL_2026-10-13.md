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
Full-model results are pending; the kernel result alone is not end-to-end proof.
