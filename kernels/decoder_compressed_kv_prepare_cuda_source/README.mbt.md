# Decoder compressed K/V preparation CUDA source

This family-neutral package emits a correctness-first CUDA source for the
borrowed-buffer boundary between already-produced ordinary/compressed BF16 K/V
rows and selected sparse attention. Prefill copies the complete ordinary
sequence followed by its completed compressed prefix. Decode copies the
physical circular-window segment followed by the completed compressed-cache
prefix. The renderer does not compute learned compression, own cache/state,
produce indices, run attention, compile CUDA, or grant launch authority.

The ABI is `(counts[5], modes[B], sequence_lengths[B], ordinary_kv,
compressed_kv, prepared_lengths[B], prepared_kv)`. Mode `0` is prefill and mode
`1` is decode. All K/V tensors are raw BF16 (`uint16_t`) and are copied without
arithmetic.
