# Fusion-cut numerical alignment

Full projection fusion and partial postprocessing must use the same numerical
schedule on either side of the materialization boundary. The compiler now owns
an immutable `AttentionIngressNumerics`: head width, strided pairwise reduction
width, Float epsilon, and AOT-materialized Float rotary basis. Head count and
fusion placement cannot change it. Normalization rounds to BF16 before paired
rotary evaluation, which rounds again before the KV commit.

A shared CUDA lowering consumes this plan in both serving artifact generators.
The partial kernel retains its 128-thread ABI but a complete subgroup performs
the pure epilogue, using exactly the full path's reduction tree. Other threads
participate in the surrounding block loads/stores and barriers. This removes
per-token `powf` and the previously different 128-lane reduction tree. Planning
does not contain CUDA names or model-family selection.

This aligns the epilogue, not the preceding projection's GEMV/GEMM accumulation.
Physical differential testing and serving measurements must distinguish these
two boundaries. Partial fusion remains an explicit offline evaluation choice
until its complete execution path has been checked; no throughput or
batch-invariance claim follows merely from source alignment.

## Implementation and validation

- `8b1f8f5`: immutable compiler numerical plan and shared CUDA epilogue.
- `d7c1a7f`, `fb5a0fc`: full-versus-materialized projection/epilogue GPU probe
  (the latter corrects the probe's canonical QKV symbol).
- `4aeaefc`: ordinary projection compilation now explicitly records its
  existing single-row subgroup reduction permission. Scalar fallback retains
  ordered accumulation. This fixes compiler metadata, not a measured speedup.

Warning-denied native check passed. The affected compiler, shared lowering,
partial/full ingress, projection and physical-probe packages passed 72 tests.
This is scoped validation, not a claim that the unrelated full repository
suite is green.

On the RTX 5060 Ti (`GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`, PCI
`00000000:17:00.0`), the old/new full kernels and the new full versus separate
QKV-plus-partial kernels matched bit-for-bit at token counts
`[1, 2, 15, 16, 17, 32, 64, 128, 256]`. The probe includes untouched output/KV
sentinels, positioned writes and a sampled independent value-projection oracle.
Its dyadic dense fixture intentionally makes dot products exactly representable;
it does not establish equivalence for arbitrary trained weights. Memcheck,
racecheck, synccheck and initcheck all passed for the complete split sequence.
The failed initial symbol lookup was retained separately, not relabeled a pass.

## Same-source Qwen serving comparison

Both servers used the kernel/runtime source at `8b1f8f5`, Qwen3-0.6B BF16,
the same GPU and existing greedy token-ID benchmark. Each cell has one warmup
and two measured trials; values below are the arithmetic mean output tok/s.
The servers ran serially, partial first and full second. Two trials do not
justify treating sub-percent differences as significant.

| Input → output tokens | Concurrency | Full fusion | Aligned partial | Change |
| --- | ---: | ---: | ---: | ---: |
| 59 → 256 | 1 | 225.2 | 232.5 | +3.3% |
| 59 → 256 | 8 | 847.3 | 971.8 | +14.7% |
| 128 → 128 | 1 | 217.5 | 222.2 | +2.2% |
| 128 → 128 | 8 | 790.7 | 892.8 | +12.9% |
| 512 → 64 | 1 | 177.3 | 179.8 | +1.4% |
| 512 → 64 | 8 | 474.5 | 501.0 | +5.6% |
| 1528 → 32 | 1 | 87.8 | 86.6 | −1.4% |
| 1528 → 32 | 8 | 126.5 | 126.0 | −0.4% |

All eight cells produce only sequences present in the full-fusion run, also
when compared with the preceding `9be6973` full-fusion baseline. Seven cells
have one unique sequence in both paths. The short 59→256 C8 cell has the same
three observed sequences in each path. This closes the **new partial-fusion
sequence divergence** on this vector; it does not establish per-request ordinal
equality or fix the pre-existing C8 batch-dependent variation. Different
GEMV/GEMM accumulation schedules remain explicitly permitted, not assumed
bit-exact by functional purity.

Partial fusion remains an explicit selection, not a universal default: these
measurements show no long-prefill gain. No other engine was rerun, and no new
SGLang/vLLM/llama.cpp speedup ratio is claimed.

## Artifacts and remaining work

Remote artifacts: `/dev/shm/lunaflux-ingress-aligned-8b1f8f5-20260906-r1`.
Successful differential/sanitizer logs:
`/dev/shm/lunaflux-ingress-cut-fb5a0fc-20260906-r2`.
Serving vectors:
`/dev/shm/lunaflux-baselines-20260906-r1-lunaflux-ingress-aligned-{full,partial}-8b1f8f5`.
Both isolated servers were stopped after measurement. Production was untouched.
The downloaded result archive is
`/private/tmp/lunaflux-ingress-aligned-20260906-results.tar.gz`; local and remote
SHA-256 both equal
`52cf6e38a78503721f57dc6415d76d7c01a760b7a8e2ea8d36c9bbc552afae34`.

Still unimplemented: distinct projection/ingress artifacts selected per bucket
at startup. Existing bucket capture varies launch geometry and attention
variants, not every projection/fusion artifact. Captured suffix arithmetic is
still predicated rather than a separately pruned graph. This follow-up does not
claim completion of those workstreams or removal of C8 sequence variation.
