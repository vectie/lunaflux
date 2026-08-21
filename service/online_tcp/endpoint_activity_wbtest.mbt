///|
test "Luna online TCP activity forbids reactor and maintenance aliases" {
  let activity : LunaOnlineTcpActivity = {
    reactor_active: false,
    maintenance_active: false,
  }
  activity.enter_reactor()
  try activity.enter_reactor() catch {
    LunaOnlineTcpFailed(LunaOnlineTcpOwnership) => ()
    _ => fail("unexpected nested Luna reactor activity error")
  } noraise {
    _ => fail("nested Luna reactor activity passed")
  }
  try activity.enter_maintenance() catch {
    LunaOnlineTcpFailed(LunaOnlineTcpOwnership) => ()
    _ => fail("unexpected concurrent Luna maintenance error")
  } noraise {
    _ => fail("maintenance overlapped Luna reactor activity")
  }
  activity.leave_reactor()
  activity.enter_maintenance()
  try activity.enter_reactor() catch {
    LunaOnlineTcpFailed(LunaOnlineTcpOwnership) => ()
    _ => fail("unexpected concurrent Luna reactor error")
  } noraise {
    _ => fail("reactor overlapped Luna maintenance activity")
  }
  activity.leave_maintenance()
  activity.enter_reactor()
  activity.leave_reactor()
}

///|
async test "Luna online TCP cancellation releases reactor activity" {
  let activity : LunaOnlineTcpActivity = {
    reactor_active: false,
    maintenance_active: false,
  }
  @async.with_task_group(root => {
    let task = root.spawn(() => {
      activity.enter_reactor()
      defer activity.leave_reactor()
      @async.sleep(10000)
    })
    @async.pause()
    assert_true(activity.reactor_active)
    task.cancel()
  })
  assert_false(activity.reactor_active)
  activity.enter_maintenance()
  activity.leave_maintenance()
}
