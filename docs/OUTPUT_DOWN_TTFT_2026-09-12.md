# Output/down operand scheduling and long-input TTFT

## Implemented compiler policy

The ordered matrix fold is unchanged. Its immutable operand window and its
independent result-row product are separate scheduling decisions:

- Long reductions (at least 2048 elements), at most 16 live rows, and at most
  two output tiles use up to eight K16 fragments per operand transfer. The
  transfer width must divide the reduction extent; other products retain four.
- A materialized down-projection consumer uses at most 32 rows per workgroup,
  instead of 64. Its sibling producer, accumulator order, output ownership,
  scalar path, and masking semantics are unchanged.
- Selected vocabulary gathers retain their separately measured window.

These are generic shape/product policies in the functional projection
compiler. CUDA transfer instructions and shared permutations remain in the
CUDA lowering. They do not add request-path allocation, validation, or model
identity branches. The profitability thresholds are conservative measured
defaults, not proof of an optimum for every device or shape.

## GPU measurements before whole-runtime integration

RTX 5060 Ti, UUID `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`, CUDA 13.1.115.
The old side is the installed `ed6f572` final-r2 runtime. Three alternating
CUDA-event graph trials compare immutable CUBINs. These are warm repeated
kernel timings, not serving throughput or cold-cache profiler timings.

| Work | Previous, us | New, us | Approximate speedup |
| --- | ---: | ---: | ---: |
| Output K2048/N1024, 8 rows | 15.01 | 13.06 | 1.15x |
| Down K3072/N1024, 8 rows | 17.10 | 12.11 | 1.41x |
| Down, 16 rows | 17.43 | 12.28 | 1.42x |
| Down, 33 rows | 41.23 | 25.79 | 1.60x |
| Down, 512 rows | 106.79 | 100.72 | 1.06x |
| Down, 1024 rows | 209.53 | 183.13 | 1.14x |

Paired output and varied-BF16 checks were bitwise equal with untouched tails.
Six representative launches passed memcheck (including full leak checking),
racecheck, and synccheck: 18 successful runs. They cover output-8 and
down-8/16/17/33/1024. Source-level excessive shared wavefronts and register
spills are zero in four new profiler captures; aggregate hardware shared
conflict totals are **not** universally zero.

Cold-cache new-kernel captures measured output-8 at 17.888 us, down-8 at
20.864 us, down-16 at 20.384 us, and down-1024 at 188.032 us. For context,
the previous report's vLLM output-8/down-8 captures were 13.152/20.256 us.
That comparison is not a fresh paired baseline run: down is close, output
still has a gap. New output-8 DRAM throughput was 53.81%, with long-scoreboard
stall share 17.42%; down-8 reached 69.27% and 37.45%, respectively.

## Rejected experiments

The broad K128-window experiment regressed short-reduction QKV by roughly
47% and the wide-output C16 map by roughly 50%. Its large matrix window also
produced an unlaunchable static/shared resource combination. A 64-row output
product was slower. Narrowing large output maps to two or four groups also
regressed the 512/1024-row workloads. A two-group, 32-row, K128 down product
improved tiny shapes but substantially regressed large shapes. None of these
regressing policies is enabled in the submitted compiler change.

## Fresh old-runtime phase trace

The unprofiled old-runtime repeat reproduced 1528/32/C8 at 298.25 output
tokens/s, mean TTFT 499.29 ms, mean decode interval 11.19 ms. This confirms
the previous roughly 501 ms TTFT observation.

A separate diagnostic parent (same worker/model/kernels) records actual
submitted rows, tokens, prefill/decode counts, and outputs. Its Nsight Systems
trace accounts for every marker without crossing kernels and exactly 256
outputs / 248 decode tokens in the long-C8 cell. Profiling timings are not
used as ordinary performance results.

The eight initial, single-request, 1024-token prefill steps average 43.01 ms
GPU span, of which 42.65 ms is kernel duration. Per step, attention is 10.79 ms,
gate/up 10.13 ms, QKV 7.25 ms, down 5.76 ms, ingress numerics 4.06 ms, and
output projection 3.78 ms. Later mixed steps take about 59–62 ms. Thus
Output/down improvements alone cannot close the long-input TTFT gap; this
trace does not support CPU dispatch being the dominant prefill bottleneck.

## Integration status

Focused native compiler and CUDA-generator tests pass (45 and 65 tests).
Whole-runtime regeneration and end-to-end measurements are pending for this
commit. No production deployment is performed or claimed by this report.
