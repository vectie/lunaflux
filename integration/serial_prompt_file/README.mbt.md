# Startup serial prompt file adapter

This package reads `plan-i.bin` once under a caller-supplied aggregate retained
byte ceiling and passes immutable snapshots to the pure
`engine/serial_prompt_frames.PreparedSerialPrompt` constructor. Each file closes
before publication; the returned prompt owns no file authority. GLM and DeepSeek
use this one implementation before opening CUDA. The constructor owns position,
identity, ordering, final sampling and total generation-capacity semantics.

No model-family branch, weight inventory, hashing, network or GPU operation is
present. This is startup preparation, not a live scheduler or per-token scan.

## Finite request queues

`load_generation_sequence` prepares every request in a finite queue before
device startup. Request i is stored under `request-i/plan-N.bin`; all requests
share one aggregate retained-byte budget. It returns immutable, model-neutral
prompt plans with no retained file authority. Capacity and byte-budget failure
in a later request prevents any request from executing.

`load_generation_sequence_file` reads the bounded `frame-counts.txt` metadata
once, then calls the same sequence constructor. Both checkpoint CLIs use this
adapter; queue parsing does not belong to a model family or device backend.
