# Physical-IR Spark regression benchmark

This diagnostic compiles the **production-generated** QKV, output projection,
selected-row vocabulary head, and gated-MLP kernels, then compares a named git
revision with the current working tree. It does not create a deployment or run
a model server. Existing `gpu_pipeline` C++/CUDA harnesses provide bitwise
before/after comparison, sampled ordered-F32 scalar referees, and five
counterbalanced CUDA-event trials. MLP reports gate/up, down, and their chain.

`prepare.mbtx ROOT REVISION baseline` creates a new baseline source checkout and
export directory. The root must already exist. After source changes finish,
`prepare.mbtx ROOT REVISION current` exports current code without modifying the
worktree. `run.mbtx ROOT` uploads only generated CUDA and harnesses to Spark
`.178` through the already authenticated SSH control socket. All output paths
are new; no model process is stopped. `prepare_ingress.mbtx` separately exports
the existing head-dimension/tail-barrier fixtures for fused-ingress validation.

After both ingress exports exist, `run_ingress.mbtx ROOT` compiles their CUBINs
and the bounded paired driver into a separate new timing campaign. It compares
heads 16/32/64/128 at 1/2/7/8/15/16/17/31/32 tokens, with three counterbalanced
CUDA-event trials of 100 launches following ten warmups. Full output and KV
buffers must match bitwise before scalar/KV reference checks and timing.
`ingress_summary.mbtx LOG_DIRECTORY` checks and summarizes all 36 cases.
`verify_download.mbtx LOG_DIRECTORY` verifies downloaded source/binary hashes
for either projection or ingress campaigns.

Safety limits:

- At least 32 GiB `MemAvailable` and no compute process before each GPU test.
- One test at a time; 8 GiB systemd memory ceiling, no cgroup swap, 64-task
  ceiling, and 180-second timeout per command.
- Explicit fixed buffer bounds independent of cgroup accounting: the largest
  head harness uses about 0.94 GB device arrays and less than 1.7 GB host arrays;
  MLP and other projections are much smaller. GPU driver allocations may not
  be fully represented by a process RSS or cgroup peak.
- No full-model profiler replay. Bounded memcheck covers QKV, output and MLP at
  a 129-token tail. Vocabulary-head memcheck is not included.

The token vector is 1/8/32/128/129/1024/2048. The vocabulary path selects up to
32 output rows, not one output row per input token. The fixed 2048-token
allocation envelope is intentional: these are kernel timings, not inference
memory efficiency or end-to-end serving results. Systemd's reported memory
peak, before/after available memory, compiler and device identity accompany
the raw measurements. Report median and range, not only the best sample.

The exporter fixture's declared specialization target remains sm120 with a
CUDA 13.1 policy. Compiling its source for GB10 sm121 with CUDA 13.0 is an
explicit fixed-schedule portability diagnostic, **not** target-specific AOT
runtime admission. A HEAD-versus-worktree delta includes prior uncommitted
changes and must not be attributed entirely to the current IR architecture.

## Attention on the second Spark

`attention_spark.mbtx prepare REPO NEW_OUTPUT 322` exports query-owned attention.
`prepare-long` uses the bounded diagnostic geometry (up to 2048 query tokens,
8192 context tokens and 4096 pages). For a baseline checkout it copies only the
current diagnostic exporter fixture, leaving the baseline production packages
unchanged. `run` or `run-long` accepts old/new export directories, a new remote
directory, a new local log directory and an authenticated SSH socket for `.179`.

The normal run includes memcheck/racecheck/synccheck. The long run uses three
paired processes with counterbalanced order and bounded arrays; it does not run
an additional long sanitizer campaign. `attention_summary.mbtx LOG_DIRECTORY`
checks successful exit statuses and summarizes medians. No full-model memory
allocation or Nsight replay occurs. See the
[2026-09-29 report](../../docs/BENCHMARK_PHYSICAL_IR_SPARK_2026-09-29.md) for
results and explicit measurement limitations.
