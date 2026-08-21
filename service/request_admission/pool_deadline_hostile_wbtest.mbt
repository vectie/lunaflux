///|
test "deadline cuts Ready before take and Prepared before claim" {
  let ready_now = Ref(11UL)
  let ready_pool = fixture_preparation_pool(ready_now, 1, 1, 4096UL)
  let ready = hostile_work(ready_pool)
  drive_preparation_ready(ready_pool, ready)
  ready_now.val = 111UL
  try ready.take_prepared() catch {
    Invalid(Deadline) => ()
    _ => fail("unexpected Ready expiry error")
  } noraise {
    _ => fail("expired Ready work published a prepared shell")
  }
  assert_eq(ready.state(), LunaRequestPreparationFailed)
  ready.discard()
  assert_eq(ready_pool.free_lane_count(), 1)

  let prepared_now = Ref(11UL)
  let prepared_pool = fixture_preparation_pool(prepared_now, 1, 1, 4096UL)
  let prepared_work = hostile_work(prepared_pool)
  drive_preparation_ready(prepared_pool, prepared_work)
  let prepared = prepared_work.take_prepared()
  prepared_now.val = 111UL
  try prepared.take_claim() catch {
    Invalid(Deadline) => ()
    _ => fail("unexpected Prepared expiry error")
  } noraise {
    _ => fail("expired Prepared shell published a claim")
  }
  assert_eq(prepared_pool.free_lane_count(), 1)
  try prepared.input_token_count() catch {
    Invalid(LunaPreparedLifecycle) => ()
    _ => fail("unexpected expired-shell replay error")
  } noraise {
    _ => fail("expired Prepared shell retained authority")
  }
}
