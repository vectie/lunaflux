# Remote worker local lease

This is a pure value state machine, not a timer or a resource owner. A runtime
replaces its current value with each result from `poll`, `renew`, or
`transport_lost`. Any status other than `Active` obligates that owner to stop
submissions, abort its communicator, and release its owned resources. Callers
must enforce that obligation; this package itself cannot terminate a worker.

All timestamps come from the same worker's monotonic millisecond clock.
Heartbeats carry generation and strictly consecutive sequence numbers, never
the coordinator's clock. Renewal at or after the deadline is too late.
Expired, disconnected, clock-invalid, and protocol-invalid states are sticky;
a late heartbeat cannot revive them. Replacement requires a separately
admitted new group generation, not another call on the old lease.

The state is a `#valtype` with immutable fields. Transition functions return a
new value, perform no I/O or hashing, and raise no hot-path error objects.

```mbt check
///|
test {
  let lease = @remote_lease.RemoteLease::start(
    generation=1UL,
    duration_millis=1000UL,
    local_now_millis=100UL,
  )
  let renewed = lease.renew(
    generation=1UL,
    sequence=1UL,
    local_now_millis=500UL,
  )
  assert_eq(lease.deadline_millis(), 1100UL)
  assert_eq(renewed.poll(1500UL).status(), Expired)
}
```
