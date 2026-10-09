# Branch integration and model preflight — 9 October 2026

## Scope and branch inventory

This first section records the initial committed-branch integration. The later
"Working-tree integration" section records the subsequent user-authorized
checkpoint and merge of previously uncommitted work; references below to
uncommitted or excluded source describe the initial snapshot, not current main.

The request named three models: GLM 5.3, DeepSeek V4 Flash DSpark, and
MiniMax H3. This work integrates committed branches and verifies whether the
installed models can actually be tested. It is not a production deployment.

| Ref at inspection | Tip | Integration |
| --- | --- | --- |
| `origin/main` | `a04f702b` | ancestor of `parallel` |
| `parallel` | `d8750c97` | fast-forwarded into local `main` |
| `origin/parallel` | `de2b4044` | ancestor of `parallel` |
| `stream` | `515c73f1` | five unique commits merged into `main` |
| `twospark` | `aad4e7e5` | already contained; no unique commits |

Integration used a separate managed worktree. The original dirty `parallel`
worktree was not reset, stashed, staged, or committed. Its untracked advanced
model-family packages and other unfinished work are **not** implied to be part
of the branch merge.

Two textual conflicts were resolved: retain both elementwise and streaming IR
imports, and retain the echo fixture's refusal to execute an approved device
runtime while adding its streaming protocol support.

## Bugs found by integration testing

- Mark the new descriptor-execution test Linux-only, following the surrounding
  tests, and explicitly execute it with `--include-skipped` on Linux.
- Use `/tmp` in the Linux process/service fixtures instead of requiring the
  macOS-specific `/private/tmp` directory.
- Make the hostile-completion fixture validate against the supervisor's retained
  submitted plan, not against the adversarial payload's own plan.
- Reject foreign prefill/decode completion slots outside the destination's
  capacity during encode preflight. Two platform-independent regressions also
  verify that rejection preserves the prior frame and epoch. This is a real
  encoder fix, not a relaxation of malformed-plan handling.
- Assert the current typed wire errors for foreign plan-count and slot limits.

## Validation

Toolchain: Moon `0.1.20260920`, moonc `v0.10.14+7d59c7ec9`; native macOS ARM64
and native Linux ARM64. The Linux run used an isolated source directory on
`.179`, a user service with an 8 GiB memory ceiling, zero swap allowance,
four-CPU quota, and a 20-minute time limit. No model weights were loaded.

The unfiltered `moon check --target native --deny-warn` still reports 142
inherited diagnostics. All 21 diagnostic files are unchanged from `d8750c97`.
Functional checks use explicit command-line exclusions `-79-25-20-29-92`;
repository warning policy is unchanged. These are not clean unfiltered release
gates.

Verified results:

- `moon info` and formatting complete; warning-excluded native check passes on
  both hosts.
- 83 worker-wire tests pass, including the two new completion-slot regressions.
- 17 pure streaming/residency/policy tests pass with **no warning exclusions**.
- The scheduler/IR allocation gate passes 10,000 streaming cycles. Rank-wire
  and socket-backed rank-child allocation gates pass with positive controls.
- Linux: 86 focused tests pass with normally skipped process tests included.
- Linux: real spawned echo-worker streaming, service reuse/cancellation,
  rank-group skew/cancellation/restore/eviction, and active-transfer failure
  cleanup/reap all pass. These exercise protocols, not CUDA/NCCL model execution.
- Streaming C probe: 16,384 ASan/UBSan scenarios pass on macOS; the identical C
  probe also passes GCC ASan/UBSan/LeakSanitizer on Linux. The stock Linux script
  initially failed because Clang was absent; GCC was used explicitly, without
  installing or replacing a system compiler.
- Separately, the original uncommitted model work passes 11 config tests and
  108 model-family AOT integration tests. These are software/reference tests,
  not inference on the installed checkpoints, and are not merged-main results.

The final parallel macOS run passed 3,517/3,518, with one `TimeoutError` in the
unchanged TCP test `zero-wait socket polls retain ready accept and fragmented
reads`. Its 2,048 empty polls have a 500 ms wall-clock deadline. The affected
package then passed 54/54 in isolation. A serial full-suite rerun is recorded
separately below; no test deadline was increased and no failure was erased.

The serial full native suite then passed **3,518/3,518** with
`--no-parallelize` and the same explicit warning exclusions. The preceding
parallel timeout remains a test-stability limitation, not a passing parallel
gate. Both macOS and Linux runs retain their earlier failed logs.

Final source-under-test tree (before this report):
`d243543b0f306b6e1f8cb685a1217407099d531d`.
Its source archive SHA-256 is
`9ad311a3980cb8c6e660abdcd9858eb34fd9f2ada733ea6f881d1d89193a6be3`.

Local logs, including initial failures and reruns:
`/tmp/lunaflux-main-integration-20261009.wNeW4J`.
Remote logs and isolated source revisions:
`/home/wlc004s/lunaflux-main-merge-20261009.3IwXX65G` on `.179`.

## Live topology verification

- Kubernetes lists `.175` and all four Sparks `.176–.179` as Ready.
- `.180` is **absent**, not registered as NotReady in this snapshot.
- H3 FL2VA and Ref2VA pods are Running on `.176` and `.177`, respectively.
  They were not stopped, redeployed, or given generation requests.
- `.178:8888` refuses connections; GLM is not serving there.
- Direct SSH and NVIDIA inspection succeed on `.178/.179`; both expose GB10
  and had no compute applications at inspection.
- Each enterprise Spark has about 121.7 GiB total unified system memory;
  available memory was approximately 117.5/116.4 GiB. GPU memory fields from
  `nvidia-smi` are unavailable on these unified-memory hosts.
- Model directories and the workspace directory exist on `.175`. A separate
  off-cluster data node and the complete NFS export path were not independently
  verified. GPU DRA pods show `Init:ImagePullBackOff`; the existing device-plugin
  pods run. No cluster configuration was changed.

## Installed models and actual test blockers

| Located directory under `/data/models` on `.175` | Directory bytes | Observed format / blocker |
| --- | ---: | --- |
| `LibertAIDAI/GLM-5.3-Flash-NVFP4` | 194,701,859,167 | `glm5_next`, ModelOpt NVFP4; merged main has no executable GLM path, and the uncommitted BF16/F32-oriented work does not supply this NVFP4 materializer/executor |
| `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` | 175,715,920,548 | EXL3 4-bit expert encoding; metadata explicitly has `serving_reader_qualified=false`; no matching LunaFlux execution path |
| `deepseek-ai/DeepSeek-V4-Flash-DSpark` | 166,898,724,734 | mixed block-FP8/packed-FP4 and DSpark-specific layers; uncommitted work still lacks a complete materialization, state, collective, and executable kernel chain |
| `MiniMax/MiniMax-H3` | 498,475,123,077 | aggregate audio/video diffusion components and variants; existing cluster services do not prove LunaFlux execution; complete LunaFlux bootstrap/worker/artifact integration remains absent |

Directory bytes are not measured peak RAM or a single all-resident H3 model.
The GLM/DeepSeek directories already exceed one Spark's total memory, so a
supported sharding/offload plan and measured reserve are required before loading.
Two idle Sparks alone do not establish working distributed inference support.

**No GLM, DeepSeek, or H3 full-model LunaFlux benchmark was completed.** Running
the config/AOT tests or querying an existing non-LunaFlux service cannot stand
in for that benchmark. The next implementation boundary is the exact installed
format plus complete executable model path, then bounded correctness and memory
tests before timing. H3 needs generation latency/denoise throughput/media metrics,
not an LLM output-tokens-per-second comparison.

## Working-tree integration

The follow-up request was to finish integration and return the original
checkout to `main`. The previously dirty source was preserved in two explicit
checkpoints on `parallel`:

- `95740551`: advanced model packages, two-node execution, parallel execution
  planning, materializers, native bindings, tests, and their documentation.
- `554a6cd9`: benchmark workload vectors and existing Spark diagnostic tools.

Main then merged these as `d5b9981a` and `fc8c415c`. The streaming branch's
host-cache/resource owners and the parallel branch's overlap/remote owners
were retained together. Startup, ownership transfer, teardown, imports, and
both streaming and event-wait sanitizer entry points were reconciled. Public
interfaces were regenerated with `moon info`. No branch was reset or deleted.
Generated build/dependency caches and benchmark results were excluded from
the commits; source and existing benchmark evidence were not discarded.

The source tested in this stage is `fc8c415c`. Its archive SHA-256 is
`a04f48f50d349208cc3ccfa325bbe43940a131637267e393f1f5973938554676`.
Linux source was extracted into the new, non-overwriting directory
`/home/wlc004s/lunaflux-integrated-main-20261009.JxxB7pol/source` on `.179`.
Local logs and the pre-checkpoint source backup are under
`/tmp/lunaflux-finish-main-20261009.H9QOGAhY`.

Completed checks at this checkpoint:

- `moon info`, `moon fmt`, native compile, and `git diff --check` pass.
- The core architecture/dependency boundary script passes.
- CUDA ordered-executor and NCCL host ABI ASan/UBSan gates pass.
- The streaming probe passes 1,024 cycles across 16 scenarios. macOS does not
  provide LeakSanitizer for this run; that limitation is not a leak-test pass.
- The benchmark harness passes its 43 focused and 50 aggregate tests.
- Linux ARM64 passes 158 targeted model-AOT, parallel-IR, remote startup,
  transport, lease, release, and group tests. These are not physical model
  inference tests. The user service limits were 8 GiB, no swap, four CPUs,
  and 20 minutes.

The first concurrent local full-suite attempts exhausted local disk space.
Their log writer also failed with `No space left on device`, so no successful
test totals can be recovered or claimed from those attempts. Only regenerable
`moon clean` build outputs were removed, freeing approximately 6.7 GiB;
original source, backups, result directories, and captured logs were kept.

The merged `fc8c415c` tree subsequently passed **4,505/4,505** tests on
Linux ARM64. This full run reached its 8 GiB cgroup ceiling but recorded
zero OOM events/kills and no swap use. The memory ceiling was not raised.

The additional tensor-parallel ownership boundary caught a merge interaction:
the streaming branch put an optional cache owner into the live resource set,
where the overlap branch requires a fully prepared ownership representation.
Commit `66ad48bb` changes this to a startup-fixed `DeviceResident` or
`Spillable` residency mode, with explicit idle/close/streaming behavior. Partial
startup still retains retryable cleanup authority. A regression verifies that
the resident mode is idle, closes idempotently, and cannot grant streaming
ownership. All eight worker tests and the unchanged ownership boundary pass.
The initial local full-suite rerun was interrupted before completion to apply
this fix; it is not counted as a pass.

The final code archive for `66ad48bb` has SHA-256
`7465b9c353bb9b8fca3379d5f1bcb1ab78fb6c4bc80a1e8526cfdb46cf9f87c2`.
Its independent Linux source directory is
`/home/wlc004s/lunaflux-main-final-20261009.tdU5mdGz/source`.

The final macOS ARM64 full native suite passes **4,506/4,506** with
`--no-parallelize`, including the added residency regression. Its output is
`final-full-tests.stdout` and diagnostics are in `final-full-tests.stderr`.
Formatting and regenerated interfaces are current. All local branches and
fetched remote branches are ancestors of the integrated `main` code revision.

On that final revision, Linux passes **179/179** focused tests, including the
rank worker, rank child and bootstrap together with the earlier 158 targets.
Both `cmd/two_spark_worker` and `cmd/tensor_parallel_overlap_rank_child` build
successfully in native release mode. Build success does not authenticate or
start either executable against a model.

The Linux warmed allocation executable also exits successfully. It exercises
both ordered and overlap policies, positive allocation controls, 65 measured
cycles per rank, zero warmed allocations/blocking synchronizations, exact
poll/submit counts, and cleanup against fake native ABIs. The enclosing shell
gate cannot complete its generated-symbol scan there because `rg` is absent;
the retained failure is a missing test tool, not a missing-symbol diagnosis.
No package was installed merely to change this result.

The complete allocation gate, including generated-symbol and fixture-order
checks, passes on macOS for `66ad48bb`. The unchanged tensor-parallel production
route boundary passes as well. An unfiltered warning-denied native check was
rerun and still fails with 142 migration diagnostics before completing the
dependency graph; normal native compilation and functional test results must
not be reported as a warning-clean release gate.

The merged source still emits inherited migration warnings. In addition,
`runtime/remote_tls/channel.mbt` uses the pinned async dependency's
`Tls::server`, which is explicitly annotated internal/testing-only. The
warning-denied check fails on that dependency alert even with the previous
migration-warning exclusions. No annotation or warning policy was weakened.
Passing functional tests does not resolve this production TLS-provider issue.

### Physical link check, not distributed model qualification

Both enterprise Sparks were idle and reachable. The verified RoCE link is:

| Node | Link address | Network interface | RDMA device |
| --- | --- | --- | --- |
| `.178` / `spark-57f5` | `10.0.0.1` | `enp1s0f1np1` | `rocep1s0f1` |
| `.179` / `spark-368c` | `10.0.0.2` | `enp1s0f0np0` | `rocep1s0f0` |

A bounded five-second bidirectional `ib_write_bw` run with 1 MiB messages,
RDMA-CM, one queue pair, and `--report-both --report_gbits` measured
106.05 and 106.06 Gb/s in the two directions. Each user service was limited
to 256 MiB and zero swap. This measures **host-memory RDMA writes**, not
GPUDirect, NCCL collectives, GPU execution, or end-to-end inference. It is
not a claim of the link's maximum possible bandwidth. The exact command and
output are in `rdma-server.*` and `rdma-client.*` in the local log directory.

### Remaining execution limits

Integrating the advanced packages does not make the installed checkpoints
executable. The installed GLM NVFP4/EXL3 formats, DeepSeek mixed quantization
and compressed/recurrent state, and complete H3 bootstrap/execution each still
need their respective complete runtime path. Two-node physical CUDA/NCCL
execution is not qualified, and the remote TLS-provider issue remains open.
The remote startup envelope also rejects the larger streaming worker schema;
merging the local streaming owner does not silently enable remote streaming.
No checkpoint was loaded, no performance comparison was fabricated, and the
H3 production services on `.176/.177` were untouched.
