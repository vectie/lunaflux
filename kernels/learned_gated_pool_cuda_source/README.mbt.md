# Learned gated pooling lowering

Five prepared effects implement reserve, joint BF16-parameter/F32 projection,
per-feature learned softmax pooling, learned RMSNorm and history publication.
The immutable semantic plan owns typed projection, gate, softmax and RMS
expressions, ordered fold seeds/extents, ratio and preceding-window overlap.
Private CUDA lowering spells those primitives and supplies launch geometry.
Existing parallel IR plans effects; the prepared frame consumes that order.
This is not GLM's mean pooling.

State persists across arbitrary contiguous chunks and singleton decode. The
previous-window first feature bank and current-window second feature bank are
combined only when overlap is requested. Pooled rows are compact, with explicit
counts and group-start positions for subsequent rotation/cache transforms.

The matching prepared frame owns retained state and bounded compact outputs.
GB10 small-fixture numeric/state, memory/leak, race and synchronization checks
pass; checkpoint binding, rotary, quantization and compressed-cache publication
remain required before model use. This is a correctness-first implementation,
not a whole-model accuracy or performance result.

Replay ports borrow eight existing materialized boundaries with live-row
metadata; normal serving inserts no diagnostic copies. Retained state is mutated
in place before history publication: rollback requires the existing snapshot
owner, not just resetting the history counter. The default source remains
byte-identical after migration. Explicit projection FMA is an alternative
program, not an equivalence claim; its small-fixture correctness and sanitizer
checks do not establish checkpoint/model quality or a performance gain.
