# Offline attention tuning

Pure parsing of measured attention observations. The exporter reads and hashes
the file once, before artifact publication. No file access occurs in execution.

The UTF-8 table has a final newline and these tab-separated records:

```
luna-attention-tuning-v1
scope<TAB>static-frontier-digest<TAB>device-id<TAB>toolchain-digest
workload<TAB>total-query-tokens<TAB>request-rows<TAB>history-tokens
record<TAB>candidate-id<TAB>median-latency-ns<TAB>sample-count
```

One or more workload records precede observations. For a workload vector, a
sample is the sum of the kernel timings over that vector, with the same vector
and repetition count for every candidate. At least three samples are required.
This selects one AOT candidate for that vector; it is not a runtime bucket
dispatch table and does not extrapolate observations to a different frontier.
Correctness must be checked before collecting a candidate's observations.

The Qwen exporter accepts `--attention-tuning ABSOLUTE_PATH SHA256 DEVICE_ID`
after its optional projection tuning arguments. Scope binds the complete
untuned frontier, including generated sources, so a compiler change requires
new measurements. Test observations are synthetic and never release inputs.
# Offline resource feedback

`parse_attention_resource_budget` is the preparatory pure pass used to
enumerate the budget-expanded frontier. `parse_attention_resource_feedback`
then matches each measurement to an actual schedule in that frontier. Unknown
register use is not zero, and measured latency still outranks resource estimates.

The native exporter accepts `--attention-resources ABSOLUTE_FILE SHA256 DEVICE`
after an optional `--attention-tuning` suffix and before `--query-metadata-v1`.
Both input files must name the same device. No live request reads these files.
The UTF-8 format ends in a newline:

```text
luna-attention-resources-v1
scope<TAB>BASE_FRONTIER_SHA256<TAB>DEVICE<TAB>TOOLCHAIN_SHA256
budget<TAB>UNITS<TAB>THREADS_PER_UNIT<TAB>REGISTERS_PER_UNIT<TAB>REGISTER_GRAIN<TAB>ALLOCATION_THREADS<TAB>SHARED_BYTES_PER_UNIT<TAB>BLOCKS_PER_UNIT
registers<TAB>SCHEDULE_SHA256<TAB>REGISTERS_PER_THREAD
```

The base frontier is exported without resource or latency inputs; this avoids
making observations define their own scope. CUDA recipes expose
`schedule_sha256` beside source identity. Resource-aware enumeration can expose
additional legal schedules; collect those separately rather than inventing
their register counts. Missing schedules remain unknown. Invalid, duplicate,
foreign-schedule and stale-scope measurements are rejected.
# Shape-specific observations

`parse_attention_bucket_routes` accepts a separate format so aggregate V1
latencies cannot accidentally become shape-specific measurements:

```text
luna-attention-buckets-v1
scope<TAB>FRONTIER-DEVICE-TOOLCHAIN
record<TAB>prefill<TAB>8<TAB>1528<TAB>256<TAB>322<TAB>91000<TAB>5
record<TAB>prefill<TAB>8<TAB>1528<TAB>256<TAB>324<TAB>77000<TAB>5
```

Fields are phase, batch rows, total query tokens, maximum context tokens,
admitted candidate ID, median nanoseconds, and sample count. The example is a
format illustration, not a downloadable production tuning table. Every measured
bucket requires the baseline with exactly matching representative shape and
sample count. Duplicate candidate/bucket rows, unknown candidates, stale scope,
and invalid phase shapes are rejected. A 1% improvement margin retains the
baseline for noise. Unmeasured buckets return no override.

The parser and immutable generic selector are implemented. Runtime multi-artifact
binding must explicitly consume the result before this counts as serving
integration; merely loading a table does not change the selected kernel.
