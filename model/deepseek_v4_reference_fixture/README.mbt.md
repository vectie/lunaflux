# DeepSeek V4 reference fixture

This package binds the architecture-neutral `advanced_decoder_reference`
oracle to the exact DeepSeek-V4 Flash-0731 semantic plan. It verifies and
exercises the plan-selected expert scoring and token-hash routes, one shared
expert combine, learned and deterministic compressed-index selection, mHC
Sinkhorn mixing, and DSpark stage order.

The fixture is bounded, deterministic, allocation-using startup/test code over
tiny `Float` arrays. It is not a checkpoint executor or production fallback.
It performs no full attention, projection, normalization, activation, weight
materialization, KV-cache, device, kernel, scheduler, or request-path work.
