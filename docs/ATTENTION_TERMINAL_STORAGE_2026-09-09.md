# Attention terminal-storage reuse

The register-resident attention output still reserved a shared accumulator
buffer for the entire fold. The compiler now gives that buffer two separate
lifetimes: a layout-probe prologue before query staging, and terminal output
after the final KV iteration. Both borrow the dead query/K prefix. Terminal
maximum and denominator borrow the dead score region instead.

This is backend-neutral lifetime allocation of a functional fold result.
CUDA lowering implements the phase boundaries with barriers. No dot, softmax,
or PV arithmetic changes in an unchanged tile schedule. Existing pairwise
live-span tests cover head dimensions 16–256 and synchronous/async schedules;
new tests cover the prologue/terminal phases explicitly.

## Physical comparison

RTX 5060 Ti, CUDA 13.1.115, BF16, 16 query heads / 8 KV heads, dimension 128,
8-token pages. Each cell is the harness median GPU time in microseconds.
Long cases use eight sequences; query counts below are **per sequence**.

| Kernel schedule | Shared bytes | Single 128, context 128 | Q128, context 4096 | Q512, context 4096 |
| --- | ---: | ---: | ---: | ---: |
| Previous 312: Q32/K32, synchronous | 45056 | 26.662 | 4657.19 | 16517.7 |
| 312 with lifetime reuse | 28672 | 26.675 | 4733.44 | 16676.5 |
| 313 with lifetime reuse: Q32/K64 | 49152 | 24.625 | 4227.13 | 15142.5 |
| 316 with lifetime reuse: Q32/K32, async | 47248 | 24.315 | 3803.85 | 13884.4 |

Storage reuse alone is not a timing win for schedule 312. It enables the
larger synchronous tile and removes the occupancy penalty of the async tile.
The portable post-allocation selector now chooses 313 for the tested high-query
compile shape: approximately 8–9% less time on the shown long cases. Async 316
is approximately 16–18% faster than the previous selected kernel, but remains
an offline-selectable alternative, **not a universally installed default**.

The campaign includes query vectors 16/64/128/256/512 and contexts
512/1024/2048/4096, plus short and ragged inputs. Unchanged Q32/K32 schedules
(both sync and async) are bit-exact against the previous output files. The
larger K64 schedule passes the independent numerical reference checks; it is
not claimed bit-exact against K32. All three new realizations pass memcheck,
racecheck, initcheck, and synccheck on the bounded correctness corpus.

A separate query-map sharding experiment was slower and was removed. Its
comparison harness also encountered duplicate log names after completing the
first comparison; that failed harness run is not called a completed campaign.

Completed campaign: `/dev/shm/lunaflux-fold-storage-20260909-r1`.
Downloaded archive: `/private/tmp/lunaflux-fold-storage-results-20260909-r1.tar.gz`.
SHA-256: `532228fb6c18b44cda16146c00e8ddd7c57be1b133c498fbb13fafe1cbf84f02`.
Folder labels are historical harness labels: c312 is old 312, c316 is new 312,
c314 is new 313, and c317 is new 316. Source defines identify the actual tiles.

These are operation-level measurements, not a fresh end-to-end Qwen result or
a new comparison with vLLM/SGLang. Broad pipeline selection and larger matrix
families remain separate work.
