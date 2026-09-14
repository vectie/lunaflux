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
