# Prepared rotary/KV frame

Owns exactly two BF16 output allocations and two AOT functions. Counts,
positions, normalized queries and shared KV are borrowed and checked before
scratch allocation. The frame returns prepared launches to the caller's
ordered executor: it submits nothing and adds no completion boundary.

Close the borrowing executor first, then this frame, then the source projection
owner. Partial preparation remains closable. Warm execution uses the borrowed
launch list without filesystem access, new heap allocation or diagnostic sync.
