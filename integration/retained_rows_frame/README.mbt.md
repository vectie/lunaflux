# Prepared retained-row ownership

`RetainedRowsFrame` borrows counts, group positions, reset, upstream error and
input rows. Startup allocates the cache, length/error state, five-word published
counts and append descriptor; only small metadata is initialized. Consumers may
read only the published row count, never the uninitialized tail.

The frame contributes three launches to the caller's queue, without creating a
stream or executor. `close` releases functions and allocations deterministically
and supports retry after a partial release. No model names or compression rules
belong in this owner. The DeepSeek adapter supplies those via an immutable plan.

The prepared ten-effect pool/transform/cache fixture passes 32 submissions with
zero measured warm heap allocations, no additional blocking synchronization and
balanced resources. This fake-device test does not establish GPU numerics.
