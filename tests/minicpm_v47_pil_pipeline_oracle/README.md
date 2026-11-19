# Full MiniCPM PIL preprocessing reference

This offline comparison tests the native processor's explicitly selected PIL
profile against Transformers **5.18.0**. The verifier downloads and retains the
original processor, backend, transform and image-utility sources and license.
It extracts their methods unchanged into a reference import/class shell. The
reference executes the original `_preprocess`, geometry, resize, normalization,
slice extraction and NaViT packing code with Pillow12 and NumPy. A small C++
CPython adapter supplies RGB8 input and serializes the resulting metadata and
Torch BF16 cast. Dependency probes are fixed to the installed reference stack;
the unused `BatchFeature` conversion wrapper returns its original data dict.
No model weights or CUDA are loaded. No reference dependency enters production.

Corpus creation, extraction, process orchestration and comparison are MoonBit.
Scratch input/pixels are overwritten only inside the new verifier output
directory, keeping disk usage independent of corpus length. On failure the
current case remains available for diagnosis.

```sh
moon run scripts/verify-minicpm-v47-pil-pipeline.mbtx ABSOLUTE_REPO CPYTHON312_ROOT TORCH_NUMPY_PIL_VENV_PYTHON NEW_ABSOLUTE_OUTPUT
```

The CPython root supplies `include/python3.12` and `lib/libpython3.12`; the
executable selects a matching venv containing Torch, NumPy and Pillow. The
recorded comparison uses Torch2.9.1 CPU, NumPy2.5.4 and Pillow12.0.0. Its report is
`target/minicpm-v47-pil-pipeline-v2/comparison/report.json`: twelve cases match
**22,682,688 BF16 words**, target sizes, slice grids, crop counts and output
shape. Cases cover source up/down/identity resizing, the448-area slicing
boundary, portrait/landscape/detail layouts, a ten-crop image and slice limits
1/2/7/9. This is sampled decoded-RGB8 preprocessing evidence, not a proof for all
images. Torchvision's alternate numeric profile, codec behavior, video input,
whole-model logits/quality and physical CUDA execution remain separate gates.
