# MiniMax VideoVAE precision correction — 2026-10-11

The installed original MiniMax H3 pipeline calls VideoVAE decoding inside an
FP16 autocast region. LunaFlux previously emitted BF16 transformer kernels.
The formats have equal storage width but different rounding and exponent
ranges; this was a real numerical-contract mismatch, not a media quality pass.

The model plan now selects an explicit `AutocastFloat::Float16` law in precision
IR. Backend-private rendering refines operand/publication primitives; model
planning does not contain CUDA types or instructions. Generic renderers retain
their existing BF16/F32 defaults for unrelated consumers. The correction is:

- post-quant convolution: FP16 operands/bias, F32 accumulation, FP16 result;
- embedding: original F32 module arithmetic, then FP16 publication, followed
  by F32 register concatenation;
- transformer: FP16 QKV, attention/projection and gated-MLP intermediates,
  retaining F32 normalization arithmetic, scales and residual storage;
- rotary: FP16 rounding after each normalized-coordinate operation and the
  angle-scale multiplication; F32 frequency multiplication and trigonometry,
  then FP16 cosine/sine and rotary products;
- final normalization/output projection: existing F32 behavior unchanged.

The original component sources were read from
`/data/models/MiniMax/MiniMax-H3/FL2VA/video_vae` on the management node. The
installed pipeline's `pipeline_minimax_h3.py` and `vae.py` provide the outer
autocast region and destandardization/decode/publication calls. Originals were
not modified.

## Actual-source tests

`scripts/probe-minimax-video-fp16.mbtx` extracts actual generated CUDA source
from the affected package test, compiles it for GB10, and checks independent
FP16 boundaries for convolution, embedding, normalization, QKV, rotary tables,
value passthrough and down projection. Remaining tail stages have finite-output
checks; this is not a complete independent oracle for their mathematical chain.
On Spark `.179`, `/tmp/lunaflux-minimax-video-fp16-20261011-v2` passes correctness,
memcheck (zero errors/leaked bytes), racecheck and synccheck. The first probe's
compile errors remain separately in `v1`; no GPU result is attributed to it.

Local native check and the complete suite pass with the existing migration
warning settings (`-20-25-29-35-79-92`): **4,994/4,994**. An earlier run had one
zero-wait socket polling failure; the affected 57-test package passes separately
and the full rerun passes. No warning setting or production hardening changed.

Checkpoint-weight decoder execution and visually/aurally meaningful joint media
validation remain required. The earlier 32×32, five-step output was finite but
showed colored patches, not a recognizable red cat. It used the old BF16 decoder
and is not a quality acceptance result. Text encoder rounding differences remain
separate from this FP16 VideoVAE correction.
