# Prepared window/compressed attention

Owns only the sliding ring, history, append descriptor and output. Compressed
values/publication and learned selections remain borrowed from their existing
owners. The immutable read-set plan selects learned rows or all causal rows;
an incompatible borrowed selection is rejected before private allocation.

Reserve, joint attention, ring copy and publication enter the containing
executor. No private queue, host readback or warm-path allocation is introduced.
Close after the containing executor closes, before its borrowed owners close.
