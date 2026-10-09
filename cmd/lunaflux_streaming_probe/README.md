# Physical streaming page probe

Run on a discrete NVIDIA GPU with physically separate host memory:

```sh
moon run cmd/lunaflux_streaming_probe --target native --release -- 0 --separate-host-memory
```

This explicit diagnostic spills a synthetic complete K/V page, restores it 128
times into another device page, compares every logical byte, preserves alignment
padding, exercises cancellation, and closes the cache, allocation and context.
The memory flag is an operator assertion for this diagnostic, not production
device qualification. Do not use it to claim added capacity on DGX Spark.

The output includes aggregate restore time and transferred bytes. This tiny
fixture is a correctness smoke test, not a representative bandwidth, TTFT,
model-token parity, or decode-interference benchmark. Those gates remain in
[the streaming plan](../../docs/STREAMING.md).
