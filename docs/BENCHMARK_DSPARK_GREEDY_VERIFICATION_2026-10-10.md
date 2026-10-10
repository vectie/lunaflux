# DeepSeek-V4 Flash DSpark: real two-Spark greedy verification

## Executed scope

Runtime source: `41650b1b8173dcc2d5aeca93d69f4f1702f16f2e` on `main`.
Clean ARM native release build completed 232 tasks under a 2-GiB/no-swap unit,
with a 543.1-MiB reported peak. Both six-row AOT stage compilations exited zero.
No checkpoint, source archive, binary or CUBIN checksum scan was performed.

The actual 43-layer checkpoint is partitioned into layers `[0,27)` on `.178`
and `[27,43)` on `.179`. Both are GB10/sm121 Sparks, CUDA 13.0:

- `.178`: `GPU-9c3d3cf0-439a-5da2-67e9-20414255879f`.
- `.179`: `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`.
- PCI on each host: `0000000F:01:00.0`.

Workload: one BOS input token (ID 0), 16 new tokens, temperature zero,
no stop token, 256 reserved context, six prepared rows. The predictor executes
all three actual `mtp.*` blocks, dependent Markov head and explicit committed
seed. Greedy target verification saves both physical states, restores/replays
accepted inputs after rejection, then commits before publishing target tokens.

The ordinary reference uses the same checkpoint, binary, placement, six-row
all-position AOT and predictor Prime callbacks, but no draft/verification loop.
It is an internal differential reference, not vLLM/SGLang or independent
upstream numerical validation. AOT launch schedules are correctness-first;
these are not the previously tuned Qwen serving kernels.

## Results

One fresh process start per mode, speculative first, ordinary second. There
are no repeated/counterbalanced trials, excluded warmups or hardware-counter
captures in this smoke. Timing starts after model preparation and includes
prompt frame IO/execution, completion rendering, generation, request release
and final token-file writing. It excludes checkpoint loading and is not an
isolated GPU decode measurement or HTTP serving throughput.

| Mode | Diagnostic generation time | Output tokens / elapsed second | Peak .178 / .179, decimal GB |
| --- | ---: | ---: | ---: |
| Actual DSpark verification | 50.562 s | 0.316 | 45.373 / 54.652 |
| Ordinary greedy target | 38.790 s | 0.412 | 48.591 / 27.153 |

Speculative completion is **30.3% slower** in this sample. Both modes produce
the exact vector:

```text
5,223,939,22,695,18752,8570,44193,24089,1237,1043,18752,3411,1227,223,1056
```

Speculative receipts report nine verified blocks, 48 submitted input rows,
15 committed input rows and eight accepted-prefix replays. Ordinary generation
reports zero prediction blocks. Thus this is real target-controlled
accept/reject generation, not just a successful draft predictor invocation.
Repeated speculative work and low acceptance are observable; a complete
latency attribution between prediction, target, backup/replay and transport
has **not** been measured. Do not assign all 30.3% to one of those components.

All four model units have `Result=success`, `ExecMainStatus=0`, no unit swap,
and no live main PID after completion. Both terminal GPU process queries are
empty. Each model unit is capped at 96 GiB, swap zero and four hours. Pure
planning includes persistent-state backup plus a 4-GiB reserve: stage estimates
are 102,097,982,664 and 74,634,052,636 bytes against 103,079,215,104-byte
ceilings. Those estimates are not measured resident peaks or proof of maximum
context capacity. Whole-model sanitizer and independent model parity remain
open; existing component sanitizer results do not cover the whole checkpoint.

## Saved raw results and harness correction

- Speculative build/AOT/run:
  `/tmp/lunaflux-dspark-run-verify-20261010-v1` (local).
- Matched ordinary reference:
  `/tmp/lunaflux-dspark-run-verify-20261010-reference-v2` (local).
- Both remote model/output roots:
  `/tmp/lunaflux-dspark-real-verify-20261010-v1` on the respective hosts.
- Clean ARM build:
  `/tmp/lunaflux-checkpoint-build-dspark-verify-20261010-v2` on `.178`.

The first reference attempt stopped before GPU launch because its local
`egress-identity.stdout` collided with the speculative receipt. That failure
and `reference-started.txt` remain in the original root. No old output was
overwritten. The successful reference uses a new local root and the same
prepared remote artifacts. The runner now namespaces both admission and
terminal GPU receipts by mode, with a standalone regression proving all eight
run/reference role names are disjoint. This fixes campaign orchestration only;
the successful model binary remains the exact committed source above.

## Remaining actual work

1. Establish independent checkpoint numerical/token parity; BOS agreement
   between two LunaFlux routes does not validate the common kernels or tokenizer.
2. Exercise real prompt chunks, more than 128 committed positions/ring wrap,
   cancellation and exhaustion on the composed two-rank checkpoint.
3. Measure prediction, target verification, state copy, replay and handoff
   boundaries on diverse text/length vectors before changing speculative policy.
4. Qualify complete GLM and MiniMax serving separately. This DSpark result does
   not finish the three-model objective or constitute production promotion.
