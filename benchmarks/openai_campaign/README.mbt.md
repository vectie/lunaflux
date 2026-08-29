# Offline Responses campaign admission

This package authenticates and replays a captured, protocol-homogeneous OpenAI
Responses comparison campaign. It deliberately does not open sockets, launch
LunaFlux, vLLM, or SGLang, read a secret, manage a GPU, or establish correctness
or comparison-pass authority.

The live-runner boundary is exact: an external owner must start the three
already-pinned engines, connect only to the declaration's `127.0.0.1:<port>`
endpoints, obtain each bearer secret through inherited descriptors (never argv,
environment, or capture bytes), sample one monotonic nanosecond clock, and write
the event stream below without post-processing. It must retain the original
`campaign.declaration.json` and all 81 `trial-NNN.capture.json` files. The replay
CLI opens those exact regular paths read-only, rejects symlink substitution,
verifies the operator-supplied declaration SHA-256, and never rewrites them.

The declaration is one duplicate-key-free JSON object with exactly these
fields:

```text
schema = lunaflux-openai-responses-campaign-declaration.v1
credential_ingress = external.inherited-descriptor.v1
credential_policy_sha256
hardware_sha256, driver_sha256, toolkit_sha256, model_sha256
tokenizer_sha256, corpus_sha256, protocol_sha256, randomization_sha256
engines[3]
profiles[9]
```

Each engine, in `lunaflux`, `vllm`, `sglang` order, contains `engine`, exact
loopback `endpoint`, and lowercase SHA-256 `revision_sha256`, `image_sha256`,
`configuration_sha256`, and independently measured exact `executable_sha256`.
Each profile, in latency, chat, long-prefill,
decode-heavy, prefix-rich, prefix-cold, saturation, churn, and mixed order,
contains `profile`, `workload_sha256`, and non-empty tokenizer-verified
`request_input_tokens` encoded as canonical decimal strings.

Each capture is one duplicate-key-free object with schema
`lunaflux-openai-responses-captured-trial.v1`, the declaration digest,
`trial_index`, `profile`, `engine`, decimal-string `ordinal` and
`order_position`, and a globally time-ordered `events` array. Each event has
exactly `kind`, `request_ordinal`, `timestamp_ns`, `input_tokens`, `status`,
`content_type`, `outcome`, and `bytes_hex`. Kinds are `submit`, `response_head`,
`sse_bytes`, and `transport_terminal`; fields irrelevant to a kind must be the
empty string or decimal string `0`. `sse_bytes` is the exact raw byte chunk as
lowercase hex. No response payload is normalized or discarded.

Run only after the external owner has closed and sealed the captures:

```text
moon run --target native cmd/openai_responses_campaign_replay -- /absolute/capture-root <declaration-sha256>
```

Successful output is branded `offline_admission_only`, binds every raw capture
digest, and states `comparison_authority=none` and
`correctness_authority=none`. It is measurement evidence for an independent
correctness/comparison verifier, never the release decision itself.

The replay executable also has an inert declaration-only `--preflight` mode.
It authenticates the exact declaration and emits the fixed engine/profile
matrix, loopback endpoints, and each engine's revision, image, configuration,
and exact executable digest. This record exists only so the separate
external-process campaign runner can require an independent live identity
observation; it starts no process and grants no identity or measurement
authority.

## Phase 4 prefix gate

The scoped `scripts/validate-phase4-prefix-benchmark.sh` validator first copies
the sealed campaign to a private snapshot. It authenticates that snapshot with
digest-pinned campaign, correctness, identity, and process-supervisor tools,
runs a copied and hash-reverified native gate executable directly, and
authenticates the same snapshot again before releasing gate output. There is no
ambient `moon run` step and no pathname reopen of the supplied campaign.

The required external Phase 4 policy is an exact, digest-suffixed 39-line
record. It pins the campaign declaration; hardware, driver, toolkit, model,
tokenizer, corpus, protocol, and randomization identities; both prefix workload
identities; every engine revision, image, configuration, and executable; both
inventories; all four verification tools; the exact gate executable and
FD-execution helper; metric; cold-regression budget;
`physical_measurement_claim=none`; and the
measurement-only authority label. The native gate joins every pin to the
authenticated declaration and handoff before replaying timestamps and all 81
correctness artifacts into the existing `BenchmarkComparison` shape. Only the
nine prefix-rich and nine prefix-cold trials are used for the Phase 4 decision;
the complete matrix remains bound by the comparison digest so losing or failed
profiles cannot be omitted.

The primary metric is aggregate completed-request rate across the three
counterbalanced trials. Equivalent submitted, completed, input-token, and
generated-token work is mandatory at each engine/ordinal coordinate. The
prefix-rich LunaFlux rate must equal or exceed the faster pinned baseline. The
prefix-cold LunaFlux rate must remain within the externally digest-pinned
regression budget of that profile's faster pinned baseline. The native parser
caps that policy value at 1,000 basis points; choosing or approving a stricter
release budget remains external policy.

Both passing and losing measured values are emitted in canonical evidence. A
losing gate exits unsuccessfully after printing the failed record. Structural,
correctness, identity, inventory, or outcome-accounting failures emit no
comparison result. The record says `physical_measurement_claim=none`,
`comparison_authority=none`, and `release_authority=none`: successful local
fixtures prove the arithmetic and trust-envelope joins, never real 81-trial
prefix parity. Build the native executable with the repository's pinned
toolchain, hash
`_build/native/release/build/cmd/openai_responses_campaign_replay/openai_responses_campaign_replay.exe`,
place that exact digest in the external policy, and supply the same executable
and digest to the validator. Compile `scripts/phase4-gate-fd-exec.c` on the
Linux evidence host with warnings denied, hash that exact helper binary, and
pin and supply its digest as well. On Linux, the narrow
helper opens and hashes the gate and policy, copies both into exactly sealed
memfds, and re-hashes the sealed descriptors before the external verifier runs.
It re-hashes them again after verification, sanitizes the gate environment,
closes the complete unrelated descriptor range, and invokes the sealed gate
with `execveat(AT_EMPTY_PATH)`. Synchronization and captured gate output use
anonymous, immediately unlinked descriptors unavailable to the verifier.
Unsupported production hosts fail closed; the
Darwin helper mode used by hostile boundary tests is explicitly test-only and
has no evidence authority. The scoped
`scripts/validate-phase4-prefix-benchmark-boundaries.sh` script exercises
source-swap, policy/tool/gate-substitution, and shell-metacharacter denial
without granting comparison or release authority.
