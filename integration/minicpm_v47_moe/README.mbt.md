# MiniCPM-V 4.7 MoE integration

`Program` owns one immutable nine-stage source/ABI/layout description. The
offline preparer exports that exact source as `moe.cu`, with stage arguments and
workspace offsets in its preparation report. The native `Frame` consumes the
same description to prepare existing ordered-executor launches. No second ABI
table or command-line workspace formula is maintained.

For the selected 32-row frame, one 1,756,224-byte workspace contains router
logits, routing indices/values, the expert map, routed intermediates, shared
intermediates and gate logits. Weights, counts, input and output are borrowed
separately. The workspace can be reused serially across all 40 layers, and must
not alias any borrowed allocation. Keep a separate arena for concurrent frames.

Startup flow:

1. Build `DecoderOperators` and `Program` outside the request loop. Compile the
   exported source to the selected target through the existing AOT artifact path.
2. Load the AOT module and layer weights through their existing owners. Allocate
   one workspace using `Program::workspace_bytes`, plus external counts/input/output.
3. Construct `Frame`, then call `prepare` with the module, weights and allocations.
   It loads nine functions, prepares all argument regions and initializes the map.
   If preparation fails partway, close the frame before releasing its resources.
4. Borrow `Frame::launches` into the existing ordered executor, or a larger decoder
   queue. Update counts/input through the caller's established transfer path.
   Execution requires no frame allocations, source generation or argument rebuild.
5. Close the executor first, then the frame's function handles, then the borrowed
   buffers and module/context through their owners. Frames never close borrowed
   allocations. `Frame::close` is also valid after partial preparation.

The native binder type-checks. Pure tests verify all nine ABIs, generated symbols,
extents, non-overlapping aligned workspace regions and source/layout consistency.
They do **not** prove CUDA compilation, actual prepared-handle cleanup or device
execution. Those remain target qualification work.

`bind_expert_weights` maps the checkpoint's packed BF16 `gate_up_proj` and
`down_proj` into slices consumed by the existing
`integration/packed_expert_mlp.upload_bank`. The source rank and axes are checked
before flattening the two leading axes. Gate/up halves stay in their source
order; down matrices retain `[hidden, intermediate]` orientation.

Bind and upload at startup, preferably within the existing checkpoint transfer
session. Upload uses bounded scratch directly into one persistent destination
bank. The adapter neither allocates a second full bank nor creates a transposed
checkpoint. It owns no GPU resources and introduces no token-step file work.

The software test transfers real safetensors bytes through the shared streaming
loader with three-byte scratch and compares each destination matrix and padding.
Actual selected-model header tests separately verify the 256-expert dimensions
and offsets. Neither test executes CUDA or loads the complete flagship payload.
