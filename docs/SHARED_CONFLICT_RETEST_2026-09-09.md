# Shared-memory counter retest — 2026-09-09

## Result

The counters are **not all zero**. Across 75 fresh profiled launches, measured
ordinary shared instructions have zero source-attributed excessive wavefronts;
remaining source-attributed excess is in LDGSTS copy instructions. Aggregate
hardware load/store conflict counters remain nonzero on several kernels,
including kernels with zero source-attributed excess. These are different
measurements, not interchangeable definitions of completion.

No production compiler or kernel changes were made during this retest.

## Source and method

- Source HEAD: `0e9aed6d8265ec4ddafafbd5ba9a6322782c0a3a`.
- Projection artifacts: `lunaflux-unified-fold-20260909-r5`; their generated
  sources were previously compared byte-for-byte against the final exporter.
- Attention was freshly exported from current HEAD, selected prefill schedule
  313: query tile 32, KV tile 64, head width 128, block 256, shared 49,152 bytes.
- GPU: RTX 5060 Ti, UUID `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`, PCI 17:00.0.
- CUDA/NCU: `/usr/local/cuda-13.1/bin`.
- NCU: SourceCounters; explicit shared load/store bank-conflict sum metrics;
  cache-control all, clock-control none, one selected launch per report.
- Source totals sum SASS `L1 Wavefronts Shared Excessive`, partitioning LDGSTS
  from other shared instructions. Hardware totals come from the raw metric CSV.
- This is a counter replay, not a fresh latency or end-to-end benchmark.

NVIDIA explains that the aggregate hardware metric can include other shared
arbitration replays; source excessive-wavefront counts better isolate address
bank conflicts. This does **not** prove the cause of every residual hardware
count in this run. See [NVIDIA's metric explanation](https://forums.developer.nvidia.com/t/shared-memory-bank-conflicts-and-nsight-metric/115731/15/).

## 1,024-token results

| Kernel | Source copy excess | Source other excess | Hardware load | Hardware store |
| --- | ---: | ---: | ---: | ---: |
| QKV | 0 | 0 | 21,478 | 8,600 |
| Output | 0 | 0 | 4,913 | 2,912 |
| Gate/up | 33,030,144 | 0 | 55,252 | 10,967 |
| Down | 0 | 0 | 5,829 | 3,724 |
| Prefill attention | 0 | 0 | 4,085 | 494,343 |

## Projection token vector

QKV K=1024/N=4096; output K=2048/N=1024; MLP hidden=1024,
intermediate=3072. All measured other-source excess is zero.

| Tokens | QKV copy | Output copy | Gate/up copy | Down copy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | shared-free | shared-free | shared-free | shared-free |
| 7 | 358400 | 179200 | 2039952 | 0 |
| 17 | 215040 | 107520 | 2043168 | 0 |
| 63 | 14336 | 7168 | 2058768 | 0 |
| 64 | 0 | 0 | 2064384 | 0 |
| 65 | 444416 | 222208 | 4102176 | 0 |
| 255 | 14336 | 7168 | 8251920 | 0 |
| 256 | 0 | 0 | 8257536 | 0 |
| 257 | 444416 | 222208 | 10295328 | 0 |
| 504 | 114688 | 57344 | 16512384 | 0 |
| 1024 | 0 | 0 | 33030144 | 0 |

Shared-free launches have hardware counts zero but do not constitute positive
coverage of a shared-memory layout. The source summarizer rejects them as
having no measured shared instructions.

## Extended coverage

Attention: tokens `[1,7,17,31,32,33,65,504,1024]`, fixed total context 1528,
query positions `1528-T..1527`, 16 query heads, 8 KV heads, page size 8.
Every source copy/other excess is zero. These all replay the **prefill**
schedule, including T=1; they do not test selected decode attention.

Default vocabulary head: K=1024/N=151936, input T=33.

| Selected rows | Source copy excess | Source other excess |
| ---: | ---: | ---: |
| 1 | shared-free | shared-free |
| 2 | 3722432 | 0 |
| 8 | 2127104 | 0 |
| 32 | 0 | 0 |

Head storage variants: K=1024/N=80, T=33, selected rows 2 and 32.
One-group strip and one/eight-group resident variants have zero source excess.
The report label `strip-g8` actually denotes the ordinary streamed eight-group
pipeline, not the specialized strip-mined plan: copy excess is 13,888 at two
rows and 21,504 at 32 rows. Other excess is zero throughout.

Small MLP: K/intermediate/output=256, T=17.

| Groups | Gate copy excess | Down copy excess |
| ---: | ---: | ---: |
| 1 | 13312 | 4928 |
| 2 | 6144 | 1792 |
| 4 | 80288 | 0 |
| 8 | 54480 | 0 |
| 16 | 41576 | 0 |

Other source excess is zero throughout this small-MLP matrix.

## Limits and next work

This retest does not cover normalization, sampling, fused ingress, selected
decode attention, all possible shapes, or other GPU backends. It cannot support
an all-kernel/all-shape zero-conflict claim.

The measured next targets are gate/up copy layout, QKV/output partial tiles,
partial-row vocabulary head, and small-group MLP copies. Ordinary consumer
source excess is already zero in this matrix. Residual aggregate hardware
counts require instruction/replay isolation before attributing them solely to
address bank conflicts. Any subsequent implementation should jointly plan
producer, shared layout and consumer in the generic compiler; this retest
does not introduce a model-specific workaround.

## Raw reports

Remote directories:

- `/run/user/1000/lunaflux-recount-0e9aed6-r1` — 44 projection reports.
- `/run/user/1000/lunaflux-recount-0e9aed6-r2` — 31 extended reports.

Each contains NCU reports, source-summary logs and raw metric CSVs. The archive
also includes the diagnostic drivers and fresh attention source:
`lunaflux-recount-0e9aed6.tar.gz`.

SHA-256: `c6cc7f067594804f0b4f95f67dc83d1601280a11be6d8c759a147527d3c8b333`.

Local download: `/private/tmp/lunaflux-recount-0e9aed6.tar.gz`.
Automation follows the MoonBit agent guide using `.mbtx` drivers.
