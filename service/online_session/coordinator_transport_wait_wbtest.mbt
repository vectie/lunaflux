///|
test "Luna coordinator preparation authenticates transport wait bound" {
  let preparation : LunaOnlineFramedCoordinatorPreparation = {
    coordinator: Some(coordinator_fixture()),
    cleanup: None,
    state: LunaOnlineInstanceReady,
    consumed: false,
  }
  assert_eq(preparation.maximum_transport_wait_millis(), 1000)
  ignore(preparation.take_ready())
  try preparation.maximum_transport_wait_millis() catch {
    LunaOnlineInstancePreparationFailed(LunaOnlineInvalid(LunaOnlineOwnership)) =>
      ()
    _ => fail("unexpected stale Luna transport wait error")
  } noraise {
    _ => fail("consumed Luna preparation exposed its transport wait")
  }
}
