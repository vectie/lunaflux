///|
test "Luna online TCP maintenance handoff is explicit and rearmable" {
  let handoff : LunaOnlineTcpMaintenanceHandoff = { pending: false }
  try handoff.require_pending() catch {
    LunaOnlineTcpFailed(LunaOnlineTcpLifecycle) => ()
    _ => fail("unexpected absent Luna maintenance handoff error")
  } noraise {
    _ => fail("off-reactor maintenance ran without a reactor handoff")
  }
  handoff.arm()
  handoff.require_pending()
  handoff.clear()
  try handoff.require_pending() catch {
    LunaOnlineTcpFailed(LunaOnlineTcpLifecycle) => ()
    _ => fail("unexpected consumed Luna maintenance handoff error")
  } noraise {
    _ => fail("consumed Luna maintenance handoff remained live")
  }
  handoff.arm()
  handoff.require_pending()
  handoff.clear()
}
