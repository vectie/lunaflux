# Worker-process end-to-end gate

This executable is built separately from `cmd/worker_echo`. The validation
script passes the exact child path to the real A/B supervisor, then proves
three monotonic plan/completion exchanges, A/B/A reuse, retained frame
inspection, clean EOF, zero child exit, and deterministic close/reap. It then
starts a replacement at predecessor 3, closes it with sequence 4 outstanding,
retires that exact recovery obligation, and proves a third child accepts and
completes sequence 5 without identity reuse.

`worker_echo` deliberately returns deterministic zero tokens rather than
opening CUDA. It proves the production framing, authentication, ownership,
process lifecycle, and sequence-safe replacement mechanics. Scheduler-side
failure publication, restart policy/backoff, and physical model execution and
device-backed readiness remain separate release gates.
