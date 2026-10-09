# Branch integration and model preflight — 9 October 2026

## Scope and branch inventory

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
