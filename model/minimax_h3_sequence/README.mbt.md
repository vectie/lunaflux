# H3 structural sequence layout

One CFG branch packs text, conditioning blocks in request order, target audio,
target video, then zero-padding to a multiple of 64 rows. Video-bearing
reference blocks place their audio before their video. Conditioning counts
refer to already encoded/projected rows, not raw media sizes. The module does
not infer encoder output geometry or admit a workflow/reference request.

`Sequence::from_request` derives target rows and hidden width from the generic
model shape and resolved request. `assembly()` supplies the explicit generic
row map directly consumable by `PackedHidden`; `segments()` retains modality
and range information for subsequent position/time metadata. `used_rows()`
distinguishes live tokens from padded rows. Budgets include padding.

`timesteps` prepares sorted distinct F32 times and row indices per evaluation.
Text/padding use video time; conditioned media clamp target time against their
noise-augmentation time. Prepare before execution, not per-step device submit.
F32-equivalent candidates share one value/index.

`schedule` prepares every evaluation and stores a single immutable row-index
and combined-AdaLN table for each distinct equality/order pattern. Per-step
values remain distinct. Explicit vision-text row overrides select video
modality for FL2VA image spans. The Int32 table-byte budget bounds unique
patterns; it does not estimate all host object overhead. No table is uploaded
or kernel launched by these pure preparation APIs.

`PositionedSequence::references` prepares Ref2VA FP64 time/height/width grids
from bounded encoded dimensions, preserving reference order, temporal origins,
spatial normalization, stereo endpoints and zero padding. Coordinates precede
rotary frequency generation and are not themselves RoPE kernel execution.

`PositionedSequence::keyframes` handles FL2VA resolved first/last anchors in
caller order without advancing target time. The last anchor uses a distinct
FP64 pairwise span policy rather than the sequential Ref2VA span.

`rotary_tables` consumes checkpoint inverse frequencies, rounds coordinates
to F32 before F32 frequency multiplication, concatenates time/height/width,
and repeats half-width coefficients for the existing attention ABI.
Both immutable tables have a combined element budget and are prepared outside
denoising dispatch. Host trigonometric results still require comparison with
the device implementation; GPU numerical equivalence is not claimed.

`attention_documents` represents noncausal visibility as disjoint row ranges:
all live modalities share one document, and padding belongs to another. No
quadratic mask is allocated. An empty padding document is omitted. The text
refiner uses only `refiner_document`, not the full packed sequence. Device
binding must honor these ranges; the existing unmasked source renderer must
not receive the entire padded buffer as a single attention document.

CFG branch composition is not implemented by this module.
Structural matching alone is not full-model numerical validation.
