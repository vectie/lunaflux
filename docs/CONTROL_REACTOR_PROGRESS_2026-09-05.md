# Nonblocking control progress

The Qwen decode profile identified a control listener's 1 ms accept/read/write
poll in the serialized inference owner's progress chain. GPU backpressure
caused a control turn on each retry, making an unrelated idle HTTP socket delay
worker completion retirement and the next token submission.

Control turns now use the existing reactor's zero-timeout I/O operation. Ready
accepts, reads and writes still progress; an empty poll returns to inference.
The existing bounded control fairness policy and singular lifecycle owner are
unchanged. No new task, native ABI, scheduler or model-family branch is added.

Polling and connection lifetime are separate policies. Each accepted connection
starts a monotonic input-idle window; received bytes refresh that window. A
response starts its bounded write window. Empty polls retain partial input and
output until the corresponding configured deadline expires. Clock failure or
reversal retires the peer. Drain and cancellation retain deterministic cleanup.
The deadline decision is a pure scalar function; clock/socket effects remain
in the transport owner, outside the compiler and scheduler.

Focused regressions cover zero-wait polling, deadline boundaries, fragmented
requests across delays greater than 1 ms, idle-client retirement, health,
readiness, malformed requests and drain. This is the production fix, not the
earlier diagnostic that disabled control progress entirely. Physical benchmark
results will be recorded separately after the commit-pinned Linux rerun.
