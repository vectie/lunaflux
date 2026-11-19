# Complete canvas sequence reference

The optional offline oracle executes the original position/grouping methods
from [Transformers 5.18.0 MiniCPM-V 4.7](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/minicpmv4_7/modeling_minicpmv4_7.py).
MoonBit downloads the pinned source/license, retains its header, and copies the
contiguous four-method span unchanged into a minimal import/class shell. A small
C++ adapter embeds CPython, supplies the actual selected configuration, and
invokes those methods through the C API. It initializes no model weights.
Corpus construction, extraction, orchestration and comparison remain MoonBit;
there is no first-party Python algorithm, test orchestration or production
Python/Torch dependency.

```sh
moon run scripts/verify-minicpm-v47-canvas.mbtx ABSOLUTE_REPO PINNED_ASSETS CPYTHON312_ROOT TORCH_VENV_PYTHON NEW_OUTPUT
```

The native adapter needs CPython 3.12 headers/shared library; the supplied Python
executable selects a venv with CPU Torch. Local evidence uses CPython 3.12.14 and
Torch 2.9.1. The report records the actual Torch version. Original source, license,
extracted method file, corpus and reference observations remain in the output.

All 154 sequences match 751,830 prompt coordinates and 1,848 appended decode
coordinates, as well as every `rope_delta` and 32-row prompt staging operation.
Cases include text, leading/trailing text, processor-derived thumbnails/detail
slices across six image geometries, adjacent frames, newline-only gaps, separate
groups, independent image/video grid streams, explicit native grids and both
4x/16x compression profiles. Some cases exceed the initial HTTP service's 4096
cache budget; they test the position owner's separate model-capacity contract.

Scope is unpadded batch-one coordinates and staging. Video labels test coordinate
grouping, not video/camera ingestion. Processor-generated crop grids are supplied
to both implementations; this does not independently verify their geometry,
pixels, image features, masked/batched padding, GPU numerics, logits or quality.
