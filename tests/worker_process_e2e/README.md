# Worker-process end-to-end gate

This executable is built separately from `cmd/worker_echo`. The validation
script passes the exact child path to the real A/B supervisor, then proves
three monotonic plan/completion exchanges, A/B/A reuse, retained frame
inspection, clean EOF, zero child exit, and deterministic close/reap.

`worker_echo` deliberately returns deterministic zero tokens rather than
opening CUDA. It proves the production framing, authentication, ownership, and
process lifecycle. Physical model execution and restart/readiness remain
separate release gates.
