# Approved tiny BF16 model physical probe

This executable is a bounded offline numerical validation fixture for the
pinned `stas/tiny-random-llama-2` revision in `tests/reference_corpus`. It
authenticates the exact config, safetensors, tokenizer, independent corpus,
compiler identity including exact bounded version components, deterministic
compile receipts, CUBINs, final execution
manifest, and streamed device weights before using the public paged production
executor with graph capture required.

The observation boundary exposes only eleven predeclared logits into
caller-owned fixed storage after a completed graph launch. It is not available
while a launch is in flight and is not imported by scheduler or serving code.
The probe checks those logits at a fixed BF16 tolerance, four greedy tokens,
and same-page KV decode behavior.

Passing this fixture supports only the tiny approved model numerical claim on
the exact admitted device/toolchain. It is not approved-model serving,
performance, arbitrary-shape, or general model validation.
