# External-process OpenAI comparison campaign

`scripts/run-openai-comparison-campaign.sh` owns the bounded host-side process
and evidence wiring missing between the authority-free benchmark adapters and
the offline Responses campaign replay. It is not part of the LunaFlux serving
runtime and adds no Python, PyTorch, vLLM, SGLang, container, or CUDA runtime
dependency.

The runner consumes one independently SHA-256-suffixed 22-line recipe. It
requires exact native executable identities for the existing campaign
preflight/replay tool, a trial driver, an external correctness verifier, a live
engine-identity verifier, and a process-group supervisor. All tools are copied
to private scratch and rehashed before execution. Credentials enter only as
already-open descriptors 3, 4, and 5 for LunaFlux, vLLM, and SGLang; no secret
path or value appears in argv, the environment, logs, or evidence.

The authenticated campaign declaration remains the source of truth for the
three revision/image/configuration/executable identities, loopback endpoints,
identical tokenizer/corpus/protocol/workload inputs, tokenizer-verified token
counts, and nine profiles. Preflight retains the declaration's distinct exact
executable digest; it never derives executable identity from an image or flags.
Before every trial, a separate pinned live-identity verifier must observe the
exact endpoint and match revision, image, configuration, and executable
identities. The trial
driver cannot self-assert that identity.

For every profile, the runner executes three trials in this Latin-square order:

~~~text
trial 1: LunaFlux, vLLM, SGLang
trial 2: vLLM, SGLang, LunaFlux
trial 3: SGLang, LunaFlux, vLLM
~~~

Every external command runs through the pinned supervisor with a bounded
timeout and cancellation grace. A successful receipt must prove exit status
zero, no timeout/cancellation, closed output streams, and an empty process
group. Each trial produces one raw, byte-preserving Responses SSE capture and
one independent correctness artifact. Only after all 81 identity, capture,
correctness, and cleanup records exist does the existing offline replay admit
the exact framing and measurement matrix.

The output is privately staged, exactly inventoried, read-only, no-overwrite,
and atomically claimed. `scripts/verify-openai-comparison-campaign.sh` requires
independently supplied tool digests, replays all correctness checks, replays the
existing campaign authority, and rejects artifact, tool, order, identity, link,
mode, or inventory substitution. Its handoff still says
`comparison_admission=external-correctness-join-required`,
`comparison_authority=none`, and `physical_measurement_claim=none`.

The external deployment/benchmark owner must still supply real pinned engine
images/executables, live identity policy, inherited credentials, an approved
process supervisor, target hardware, real correctness/reference logic, and the
named reviewer that joins these captures to `benchmarks/evidence` comparison
authority.
