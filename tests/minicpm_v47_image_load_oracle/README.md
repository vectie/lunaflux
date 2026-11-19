# MiniCPM image loading reference

This offline reference invokes unchanged `load_image` from Transformers5.18.0
with Pillow12. It covers126 PNG/JPEG cases, all eight EXIF orientations, both
TIFF byte orders, thin/odd/block-boundary dimensions and several color modes.
Python and Pillow are test dependencies only. Downloaded source and its license
are retained in the output directory.

```sh
moon run scripts/verify-minicpm-v47-image-load.mbtx REPO PYTHON_ROOT PYTHON_EXECUTABLE NEW_OUTPUT
moon run scripts/diagnose-minicpm-v47-jpeg.mbtx REPO NEW_OUTPUT/comparison NEW_DIAGNOSTIC_OUTPUT
```

`target/minicpm-v47-image-load-v3/comparison/report.json` records exact EXIF
orientation agreement across188,790 bytes when the reference starts with the
same native-decoded RGB raster. All72 PNG cases also match complete Pillow
loading. Across54 JPEG cases,14 are bit-exact and the maximum channel difference
is3. These are sampled codec results, not a general error bound or model-quality
acceptance.

The JPEG diagnostic isolates decoder arithmetic from orientation. It exposed
the vendored decoder dropping buffered progressive scan bits after prefetching
a marker. The corrected fixed-size corpus reduces maximum error58 to2; the
expanded corpus measures3. Default SIMD and scalar builds give identical
results. Scalar mode is diagnostic only; production retains SIMD. Timings are
small host codec measurements and establish no GPU/model performance claim.

The standalone RGB8 ASan/UBSan probe exercises the production correction,
independent progressive raster fixture and truncated input prefixes with
balanced decoder allocation counters. It disables leak detection. No model or
accelerator runs in this oracle.
