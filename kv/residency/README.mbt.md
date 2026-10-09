# Bounded residency

Pure state transitions control preallocated slot storage. Each transfer carries
an owner, slot generation and operation epoch. Cancellation retains its credit
until completion. Failed transfers require a successful backend drain before
reclamation. This package owns metadata only and imports no device backend.

```mbt check
///|
test "a spill becomes reusable only after completion" {
  let limits = @residency.Limits::new(
    slots=2,
    transfers=1,
    bytes_per_slot=4096L,
    host_budget=8192L,
    separate_host_memory=true,
  )
  let pool = @residency.Pool::new(limits)
  guard pool.begin_spill() is Started(copy) else { fail("admission") }
  assert_true(!pool.is_host_ready(copy.entry()))
  // A trusted effect interpreter reports actual DMA completion here.
  assert_true(pool.complete(copy) is HostReady)
  assert_true(pool.is_host_ready(copy.entry()))
  assert_true(pool.evict(copy.entry()) is Released)
  assert_true(pool.close())
}
```

Transfer credits and host slots are independent capacities. Full slots or busy
lanes return `Backpressured`. Stale and foreign completions return `Rejected`
without mutation. Generation exhaustion retires a slot; epoch exhaustion
requires eviction rather than wrapping. No normal transition allocates.
