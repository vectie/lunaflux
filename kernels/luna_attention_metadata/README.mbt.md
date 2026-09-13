# Attention step metadata

Portable storage layout for query tiles of 16, 32, 64 and 128 rows. The first
four Int32 cells hold the live record counts. Each bucket reserves
`maximum_rows + (maximum_tokens - maximum_rows) / tile_width` records.

Each eight-cell record contains request row, token begin, request-row begin/end,
sequence length, page-table begin/end, and exclusive key end. The runtime
publishes records for prefill rows once per step; decode rows are excluded.
All attention heads and layers consume the same immutable publication.

`bucket(width)` is the checked lookup. `bucket_offset(ordinal)` is the bounded
owner-loop primitive: callers supply an ordinal in 0 through 4 (4 is the end
offset). Device-step publication uses the fixed 0-through-3 loop. This package
contains no CUDA types, allocation ownership, filesystem access or tuning logic.

The metadata ABI is explicitly versioned in runtime bundle v5. Its consumer
must not interpret this storage as the older CSR row-offset array.
