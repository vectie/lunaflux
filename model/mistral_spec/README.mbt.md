# Bounded Mistral semantic specification

`model/mistral_spec` owns validated, content-bound metadata for the exact
`MistralForCausalLM` subset that can be expressed by LunaFlux's existing dense
operation vocabulary. It validates BF16 dimensions through the established
dense contract and records both the declared position ceiling and the optional
sliding window.

The public context ceiling never exceeds one sliding window. The type owns no
plan, tensor bytes, filesystem capability, device resource, or scheduler state.
