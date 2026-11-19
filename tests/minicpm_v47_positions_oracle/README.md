# Offline canvas coordinate oracle

This test executes the same ATen FP32 `linspace`, `round` and integer conversion
used by Transformers 5.18.0's
[`MiniCPMV4_7Model.get_vision_position_ids`](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/minicpmv4_7/modeling_minicpmv4_7.py).
The MoonBit executable compares thumbnail coordinates through the public
`Sequence::build` API, stretching them over one detail crop on both spatial axes.
Counts1–64 and extents1–128 plus four larger endpoint cases give 8,196 cases and
533,500 compared coordinates. Synthetic grids intentionally isolate resampling;
they do not represent a corpus of camera images or independently validate slicing.

```sh
moon run scripts/verify-minicpm-v47-positions.mbtx ABSOLUTE_REPO LIBTORCH_ROOT NEW_OUTPUT
```

The LibTorch root contains `include/` and `lib/`; a C++17 compiler is required.
Only the offline oracle links LibTorch. The production position package remains
pure MoonBit, with no Torch or Python dependency. The report records the actual
Torch version. Local evidence uses LibTorch2.9.1 CPU. Full upstream processor,
CUDA/Spark arithmetic and model quality remain separate gates.
