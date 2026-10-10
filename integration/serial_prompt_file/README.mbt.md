# Startup serial prompt file adapter

This package reads `plan-i.bin` once under a caller-supplied aggregate retained
byte ceiling and passes immutable snapshots to the pure
`engine/serial_prompt_frames.PreparedSerialPrompt` constructor. Each file closes
before publication; the returned prompt owns no file authority. GLM and DeepSeek
use this one implementation before opening CUDA. The constructor owns position,
identity, ordering, final sampling and total generation-capacity semantics.

No model-family branch, weight inventory, hashing, network or GPU operation is
present. This is startup preparation, not a live scheduler or per-token scan.
