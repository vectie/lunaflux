# Diagnostic worker execution trace

Run `install_execution_trace.mbtx DISPOSABLE_SOURCE` only against a disposable
source copy. Rebuild `cmd/device_worker_child` and `cmd/lunaflux`, and bind both
executables in a new diagnostic launch. The diagnostic parent retains inherited
stderr instead of closing it before worker exec. The normal repository packages
do not import these hooks or change their descriptor/environment isolation.
No runtime flag, filesystem lookup, or diagnostic FFI call is added to production.

The installer checks exact source seams. All FFI arguments are primitive values;
the C sink uses a bounded stack buffer and synchronous stderr writes. Therefore
this trace is for attribution, not an uninstrumented throughput claim. stderr
must be retained by the harness, not interpreted as the native worker protocol.

Every `lf_exec` line contains a monotonic nanosecond timestamp, kind, sequence,
six integer fields (`a`–`f`), and elapsed nanoseconds:

| Kind | Sequence | a–f |
| --- | --- | --- |
| 0 | Wire sequence | rows, tokens, prefill rows, decode rows, 0, 0 |
| 1 | Wire sequence | prefill row index, query length, past-token count, 0, 0, 0 |
| 2 | Wire sequence | decode row index, 1, past-token count, 0, 0, 0 |
| 3 | Descriptor epoch | bucket slot, final owner, bucket rows, bucket tokens, bucket context, output rows |
| 4 | Descriptor epoch | submitted owner, captured=1/eager=0, 0, 0, 0, 0 |
| 5 | Descriptor epoch | completed owner, 0, 0, 0, 0, 0 |

Kind 3 elapsed measures bucket and initial phase-owner selection, excluding row
trace writes and descriptor publication. The reported owner is the **final**
owner after output-demand selection. This interval does not claim to include
the later output-demand selection. Correlate wire sequences and epochs in event
order; they are different domains and must not be assumed numerically equal.

Bucket upper bounds are capacity, **not executed padded FLOPs**. Use per-row
query/history and the selected kernel's actual launch geometry and tile plan
to derive padding. Correlate kinds 4/5 with Nsight Systems CUDA graph nodes to
separate device execution, completion waits, and inter-step GPU gaps. Do not
subtract diagnostic stderr cost from end-to-end timing without measuring it.

## Match logical work across frameworks

`run_matched_trace_campaign.mbtx` accepts `--output 64` or `--output 256`
and diagnostic-only `--work-rows INSTRUMENTATION_ROOT`. The instrumentation
root holds `install_execution_trace.mbtx`, `vllm-work.py`, and
`sglang-work.py`; generate the reference overrides from pinned source
copies with `install_reference_work_rows.mbtx`. Never install them into a
production container. Reference markers use existing CPU sequence metadata,
not device-to-host reads or synchronization.

Run `summarize_work_shapes.mbtx TRACE_ROOT ENGINE CLIENT_JSON NEW_OUTPUT`
after the capture, once per framework.
It joins CUDA launch correlations to CPU row markers and reports sorted
`query:past` vectors with their multiplicity. Equal grids do not imply equal
work: capacity CTAs may exit, and different engines use different prefill
chunk sizes. Keep unmatched launches visible. Marker traces are attribution
runs, not ordinary throughput measurements.

`profile_matched_prefill.mbtx` checks the previously captured logical work
before replaying the selected prefill invocation. Its reference operands are
live, whereas LunaFlux's standalone probe uses deterministic synthetic
operands; this is a matched-shape comparison, not identical operands.
`profile_prefill_ablation.mbtx` instead captures both modules from a saved
paired probe command on identical operands. Counter replay times must remain
separate from unprofiled serving time.

`prepare_affine_serving.mbtx` qualifies an isolated compiler artifact from a
frozen runtime plus a scoped compiler overlay. It exports the actual compiler
candidate, checks the source and functional identity, compiles deterministically,
checks pure/mixed/tail outputs and sanitizers. Next,
`recalibrate_affine_serving.mbtx FROZEN_SERVING QUALIFIED_AFFINE MEASURE_TOOL
CALIBRATION NEW_EMPTY_ROOT` replaces only that prefill module, re-exports the
bundle and measures fresh module-bound routes before materializing serving.
All other modules and workers stay frozen. Old route observations cannot be
relabeled for a changed module set. The normal source retains the checked view
for arbitrary positions.
