# Prepared learned pooling

Owns two F32 retained windows, history, projection frames and compact normalized
output/count/position buffers. Parameters and the containing decoder executor
are borrowed. Startup initializes retained KV to zero and scores to negative
infinity with bounded 64 KiB host chunks. Token steps allocate nothing and
submit no additional queue or host synchronization.

Close the containing executor before this frame. The append descriptor exposes
sticky position/control errors to that executor. Checkpoint binding, output
rotation/quantization and compressed-cache append remain separate integration.
