---
name: lunaflux-ako
description: Optimize LunaFlux GPU execution using an AKO-style measured loop. Use for kernel/compiler performance work, route tuning, or benchmark-driven optimization in LunaFlux; not generic feature development or deployment.
---

# LunaFlux AKO for Codex

Adapted from AKO4ALL's profile → modify → measure → record loop. This is an
offline developer workflow, never a dependency of the production runtime.

## Resolve the experiment

Read the repository AGENTS.md and its named architecture contracts. Identify
the actual selected artifact/symbol, workload vector, numerical contract,
reference, affected compiler pass, GPU, and output directory. State these
choices briefly. Resume existing experiments instead of resetting history.

Keep existing compiler boundaries: semantic/numeric program → legal schedules
→ resource/measured selection → ownership/storage/effects → backend lowering.
Change a pure plan or pass when the optimization is general; change CUDA only
for terminal device-specific lowering. Do not introduce model-name branches,
request-path JIT, profiler checks, or benchmark dependencies in production.

## Run the loop

1. Reproduce a paired baseline on the exact workload. A probe must use runtime
   launch geometry and include all launches of a fused/partitioned chain.
   Record artifact hashes, GPU identity and correctness. Check real dispatch
   separately: a new source file does not prove serving selected it.
2. Use hardware counters and source/SASS to form one falsifiable hypothesis.
   Distinguish excessive work, dependencies, copies, synchronization and host
   gaps. If counters are unavailable, label the explanation unverified.
3. Implement one coherent change. Generate and compile AOT artifacts offline.
   Reuse frozen baselines; do not repeatedly rebuild unchanged references.
4. Measure unprofiled paired runs, alternating order. Keep inputs, numerical
   tolerance, trial counts and timing boundaries fixed. Retain failures and
   regressions. Run boundary correctness and the changed sanitizer/leak gate.
5. Record hypothesis, changed files/pass, raw output paths, selection identity,
   all workload deltas, correctness and decision before the next experiment.
   Commit only owned files, not `git add -A`. Do not overwrite or restore a
   dirty worktree. Unsuccessful experiments remain diagnostic, not production.
6. Verify the winner end-to-end with selected symbols/owners. Include TTFT,
   output-token throughput and per-request token/timing vectors. Report losing
   shapes too. Kernel speedup is not whole-engine speedup or quality parity.

Use `benchmarks/gpu_pipeline/ako_trial.mbtx` for bounded paired probe trials.
Its JSON contract names an existing executable and fixed workload arguments;
it does not invent correctness or authorization. Read the script's tests and
schema before constructing a contract. Existing serving/profile campaign
helpers remain authoritative for end-to-end and sanitizer measurements.

## Constraints and stopping

All agent-authored automation is MoonBit `.mbtx`. Do not copy Ako's Python/
PyTorch evaluator, shell wrapper, language-switching policy, broad permission
settings, whole-tree staging, or destructive best-commit restoration.

Serialize GPU workloads per device. Bound memory and runtime externally with
the host's user-systemd/container mechanism. On unified-memory machines,
preserve the configured MemAvailable reserve in addition to process limits.
Do not embed credentials in skills or experiment files. Do not change global
Codex permissions or launch more agents without authorization.

Use a finite experiment budget stated at the start. After repeated non-wins,
reprofile and change the hypothesis; do not make success the only termination
condition. Stop and report preserved failures when infrastructure or accuracy
prevents a valid result. A best-so-far result is not proof of optimality.

See `docs/AKO_CODEX_LUNAFLUX.md` from the repository root for adaptation details
and the measured pilot, and `docs/BENCHMARK_LONG_CONTEXT_2026-10-04.md` for the
long-context baseline. Upstream: `../ako`, revision
`bbd0e1cf1ce2fb19d4322a932680f4c3d175d80e` (MIT, TongmingLAIC 2026).
