# Streaming rank process integration gate

On native Linux, build `tests/streaming_rank_echo` and run this package with the
absolute path to that executable. It spawns two real child processes and checks
skewed completion, cancellation after one shard completes, subsequent restore,
eviction, and healthy close. Add `--failure` after the path to fault an active
exchange and verify cleanup and reap retain the pending ownership fence.

The fixed fixture metadata exercises protocol and resource lifetime behavior.
It supplies no CUDA payload, NCCL correctness, model parity, or performance
evidence.
