///|
test "phase23 balanced non-batched scheduled cancel remains legal" {
  let ledger = Phase23SoakLedger::new(8)
  phase23_soak_record_balanced_wave(
    ledger, true, false, true, true, true, false, false,
  )
  assert_eq(ledger.two_live_waves, 1UL)
  assert_eq(ledger.batched_waves, 0UL)
  assert_eq(ledger.backpressured_waves, 1UL)
  assert_eq(ledger.scheduled_cancel_waves, 1UL)
  assert_eq(ledger.attempted_cancel_waves, 1UL)
  assert_eq(ledger.accepted_cancel_waves, 0UL)
  assert_eq(ledger.cancelled_waves, 0UL)
}
