# Physical benchmark qualification adapter

The `benchmark` campaign drives the existing digest-pinned spawned runtime and
native TCP listener. It records lifecycle timestamps with
`BenchmarkTrialCollector` at the real frame-write, authenticated `Accepted`,
checked first-token, and checked `Completed` boundaries. The campaign publishes
the canonical raw-event and summary bytes, their independent SHA-256 digests,
the release pins, exact request/network/KV outcome, and a digest of the complete
record only after deterministic owner cleanup.

Build and run it on the provisioned target using the same materialized launch
and independently computed worker digest as the physical execution campaign:

```sh
moon build --target native --deny-warn tests/approved_model_spawned_physical
_build/native/debug/build/tests/approved_model_spawned_physical/approved_model_spawned_physical.exe \
  benchmark \
  ABSOLUTE_DEPLOYMENT_ROOT#sha256=LOWERCASE_LAUNCH_SHA256 \
  LOWERCASE_CANONICAL_CHILD_SHA256
```

This is a measured LunaFlux qualification trial for the pinned one-input-token,
two-output-token request. It is not the documented 128/128 latency workload,
does not include warm-up requests, and is not admitted as a vLLM/SGLang
comparison. Full comparison still requires pinned baseline adapters, identical
tokenized corpora and workload configurations, all nine profiles, three
counterbalanced trials per engine/profile, correctness digests, and publication
of failures and losing workloads.
