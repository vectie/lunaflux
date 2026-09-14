# Attention compiler CPU benchmark

Run `moon run tests/attention_compiler_bench --target native --release`.

Measures complete functional frontier construction versus in-process reuse of
the same 12-candidate query-owned request (2048 queries, 4096 context). Each arm
checks full result equality; the reuse arm also checks all 12 hits. One warmup
per arm precedes five alternating-order trials, 20 compilations per trial.

This includes equality-check overhead. It excludes source emission, external
CUDA compilation, cold process startup, file cache and all GPU execution.
It is not a performance threshold in the test suite. Resource-changing,
shape-changing and persistent-cache workloads require separate measurements.
