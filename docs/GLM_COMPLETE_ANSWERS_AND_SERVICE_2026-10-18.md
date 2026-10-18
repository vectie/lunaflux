# GLM complete-answer and real-service acceptance

The previous six-prompt, eight-output-token timings are not a quality corpus.
This campaign uses independent expected answers/task checks, not reference
engine tokens. All evaluation remains outside the compiler and runtime.
No payload hashes, tolerance changes or production promotion are introduced.

## First campaign

Use both GB10 Sparks as TP2, 64-row prefill capacity, context1024 and at most512
new tokens. Re-export the actual AOT for that context rather than relabel the
old64-context artifact. Plan under100GiB per rank with its existing2GiB internal
reserve, externally cap each process at100GiB/no swap, and stop both ranks if
either host falls below16GiB MemAvailable. No overlapping GPU workloads.

Freeze a broader corpus before inference: factual short answers, multi-step
arithmetic, JSON/array formatting, code, multilingual translation, supplied-text
retrieval, longer distractor context, and reasoning. Retain complete raw tokens,
reasoning and final text, finish reason, input length and runtime selection.
An exhausted output budget is inconclusive; increase the admitted envelope and
rerun that case rather than count a truncated answer as correct. Do not silently
strip malformed framing or count reasoning-only text as the final answer.

Begin with the retained ordered-branch control, then the overlap candidate on
the identical corpus and envelope. Repeat relevant failures and isolate model
errors from request-state or decoder presentation errors. Stable wrong answers
are not fixed by copying another engine; semantic/state violations require a
source fix with regression tests. Code execution must be separately sandboxed
and bounded; never execute generated code inside the serving runtime.

## Service gate (not the CLI gate)

The resident serial CLI provides an initial model-output check only. The public
endpoint stays offline until the existing full HTTP/scheduler path is wired to
the batched GLM TP owner and validated on both GPUs. Required coverage includes
complete JSON/SSE answers and their agreement, multiple clients, mixed/chunked
prefill and decode, seeded sampling, cancellation/disconnect and slot reuse,
capacity rejection/recovery, and failed-rank drain. Echo-worker transport tests
and CPU tests do not establish this real-model acceptance.

The memory envelope must be measured for the actual context/concurrency plan;
the first1024-context diagnostic is not a declared production capacity limit.
Do not restore the old GLM containers or publish an incomplete replacement.

## Generation-control correction before running

The installed checkpoint declares EOS IDs154820,154827,154829. The old serial
CLI admitted only one stop ID and its inference limits allowed only one. It now
accepts an explicit bounded comma-separated set, preserving the old single-ID
and `-1` forms. The complete-answer harness supplies all three declared IDs.
Each retired result writes a finish reason and actual token count separately
from its raw tokens. This is a generation-control fix, not a numerical-policy
change or evidence that earlier answers actually crossed an ignored EOS.

Local warning-denied native check passes with existing migration exclusions.
The command package's3 tests and serial-generation package's11 tests pass;
the extended offline runner passes5 tests. Interface generation reports1096
pre-existing migration warnings and zero errors. These are preflight checks,
not physical-model or full-service acceptance.
