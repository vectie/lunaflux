# Digest-pinned OpenAI baseline observations

This package bridges OpenAI-compatible LunaFlux, externally operated vLLM, and
externally operated SGLang servers into LunaFlux's authority-free benchmark
collector. It does not launch a process,
open a socket or file, read credentials, own a container/device, or depend on
either engine. The external campaign owner retains those responsibilities and
calls `submit` immediately before its request write, passes a normalized HTTP
status/content-type observation, and feeds response bytes into the bounded SSE
decoder as they arrive.

The package has separate Chat Completions and Responses `text/event-stream`
observers. Their observed protocol identities are closed enum values with
canonical digests derived from `openai.chat.completions.sse.v1` and
`openai.responses.sse.v1`; a caller-supplied request-protocol digest is
recorded separately and cannot relabel measurements. HTTP 200 is
the admission equivalent. Every non-empty assistant content delta must carry
token-level `logprobs.content`; the first such token is the first-token
observation, and the exact observed token count must equal terminal usage. A
checked `stop`/`length` finish, exact usage record, and `[DONE]` form the
completed terminal observation. Multi-choice,
tool/function/reasoning output, unknown finish reasons, duplicate JSON keys,
missing or inconsistent usage, content after finish, premature or duplicate
`[DONE]`, content without token logprobs, non-SSE success responses, and
unclassified HTTP statuses fail closed. Transport timeout, cancellation, or
failure remains accountable after content because the required token-logprob
entries provide an exact generated-token count even without terminal usage.

Every trial preallocates one fixed event buffer per declared request and uses
the startup-fixed `BenchmarkTrialCollector` lanes. Each engine identity binds
revision, image, and engine-configuration SHA-256 values. The comparison
engine's executable/container digest is a canonical
composite of image plus engine configuration, so flags cannot disappear when
the result becomes a `BenchmarkEngineIdentity`.
Shared input declarations bind
profile, tokenizer, corpus, protocol, workload configuration, and every
tokenizer-verified request input length; LunaFlux, vLLM, and SGLang
declarations must compare identically before a campaign begins. The separate
native-framed LunaFlux qualification trial is intentionally not accepted by
this OpenAI comparison path.

`OpenAIBaselineTrialEvidence` and `OpenAIResponsesTrialEvidence` expose
measurements, immutable source and comparison identities, and workload
declarations. The branded
`OpenAIComparisonMeasurementSet` admits exactly 81 observations in the shared
Chat Completions SSE mapping: three engines, nine profiles, and three
counterbalanced trials. `OpenAIResponsesComparisonMeasurementSet` admits the
same exact matrix only from the actual Responses lifecycle observer. A
native-framed LunaFlux trial cannot inhabit either set.

Neither type contains a `BenchmarkTrial` or correctness-passed authority. An
independent verifier must join these measurements with real correctness
evidence before the existing comparison admission can be called. This package
therefore establishes neither correctness nor a physical performance result.
The Chat set cannot satisfy the production `/v1/responses` observation
boundary: its `require_responses_v1_observation` always refuses. The Responses
set satisfies only that protocol-observation boundary and still grants no
correctness or comparison-pass authority. Captured-timestamp observers are
additionally branded with a distinct external-capture measurement digest, so
offline replay cannot masquerade as an in-process live-clock measurement.
