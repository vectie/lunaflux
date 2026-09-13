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
