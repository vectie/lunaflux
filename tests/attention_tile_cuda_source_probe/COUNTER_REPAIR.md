# Counter-repair probes

`benchmarks/gpu_pipeline/export_counter_repair.mbtx` exports actual compiler
sources, without rewriting the generated kernels. Prefill variants cover the
KV128 sync/async domain and ingress variants cover independent CTA row sizes.
`check_counter_repair.mbtx OUTPUT_ROOT CUDA_ARCH [decode]` runs on the CUDA
host, sequentially, preserving logs and refusing to start a command when
available memory is below 32 GiB or another compute process is present.
It uses fixed small diagnostic allocations, not model loading.

`owned_decode_probe.cu` is compiled with one exported grouped-decode source
using `-include`. Its head counts, page layout and capacity match the export
fixture. The block size comes from exported launch metadata, not the generated
kernel's private macros (which are undefined at the end of the source).

The oracle covers:

- key-tile tails and repeated double-slot reuse;
- mixed prefill/decode rows and C1/C2/C8;
- a double-precision CPU softmax reference, tolerance 0.004;
- unchanged persistent K/V and untouched prefill output;
- invalid cache pages, whose transfer-validity exit must leave output untouched;
- 4096-token history outside the bounded sanitizer subset.

Run memcheck, racecheck and synccheck separately. A source-fixture pass is not
model qualification, a serving benchmark, or proof that the new schedule wins.
