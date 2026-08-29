///|
fn wb_publish_multi_accepted(instance : LunaOnlineInstance) -> Unit raise {
  let effective = try! @inference.EffectiveLimits::new(
    max_new_tokens=2,
    context_ceiling=8,
    instance.inference,
  )
  try! instance.events.publish_accepted(instance.expected_model, effective)
  instance.mark_multi_event_published(0, AcceptedEvent)
  assert_true(instance.events.has_event())
  assert_eq(instance.multi.event_lane, 0)
  try! instance.events.discard()
  instance.multi.event_lane = -1
}

///|
test "multi accepted publication binds each fresh instance exactly once" {
  let first = coordinator_fixture_instance()
  wb_publish_multi_accepted(first)
  let second = coordinator_fixture_instance()
  wb_publish_multi_accepted(second)
}

///|
test "multi disconnect distinguishes live and already-terminal event stages" {
  let instance = coordinator_fixture_instance()
  let lane = instance.multi.lanes[0]
  lane.phase = LunaRequestActivePhase

  instance.event_kind = AcceptedEvent
  assert_false(instance.finish_multi_abort_after_event_cut(0))
  assert_true(lane.phase == LunaRequestActivePhase)
  assert_eq(instance.multi.retirement_lane, -1)

  instance.event_kind = TokenEvent
  assert_false(instance.finish_multi_abort_after_event_cut(0))
  assert_true(lane.phase == LunaRequestActivePhase)
  assert_eq(instance.multi.retirement_lane, -1)

  instance.event_kind = CompletionUsageEvent
  assert_true(instance.finish_multi_abort_after_event_cut(0))
  assert_true(lane.phase == LunaRequestReleaseReadyPhase)
  assert_eq(instance.multi.retirement_lane, 0)

  lane.phase = LunaRequestActivePhase
  instance.multi.retirement_lane = -1
  instance.event_kind = CompletedEvent
  assert_true(instance.finish_multi_abort_after_event_cut(0))
  assert_true(lane.phase == LunaRequestReleaseReadyPhase)
  assert_eq(instance.multi.retirement_lane, 0)

  lane.phase = LunaRequestActivePhase
  instance.multi.retirement_lane = -1
  instance.event_kind = FailureUsageEvent
  assert_true(instance.finish_multi_abort_after_event_cut(0))
  assert_true(lane.phase == LunaRequestReleaseReadyPhase)
  assert_eq(instance.multi.retirement_lane, 0)

  lane.phase = LunaRequestActivePhase
  instance.multi.retirement_lane = -1
  instance.event_kind = FailedEvent
  assert_true(instance.finish_multi_abort_after_event_cut(0))
  assert_true(lane.phase == LunaRequestReleaseReadyPhase)
  assert_eq(instance.multi.retirement_lane, 0)
}

///|
test "routed terminal wins a stale worker-failure cut" {
  let cancelled = coordinator_fixture_instance()
  let cancelled_lane = cancelled.multi.lanes[0]
  cancelled_lane.terminal_cut = WorkerFailureCut
  cancelled_lane.failure_kind = WorkerLatchedFailure
  cancelled_lane.recovery_close = true
  cancelled.multi_prepare_worker_failure_race_terminal(
    0,
    OnlineCancelledByCaller,
  )
  guard cancelled.event_kind is CompletionUsageEvent else {
    fail("cancelled terminal was rewritten as worker failure")
  }
  assert_true(cancelled.events.has_event())
  assert_eq(cancelled.multi.event_lane, 0)
  assert_true(cancelled_lane.terminal_cut == NoTerminalCut)
  assert_true(cancelled_lane.failure_kind == NoLatchedFailure)
  assert_true(cancelled_lane.finish_kind is FinishCancelled)
  assert_false(cancelled_lane.recovery_close)

  let deadline = coordinator_fixture_instance()
  let deadline_lane = deadline.multi.lanes[0]
  deadline_lane.terminal_cut = WorkerFailureCut
  deadline_lane.failure_kind = WorkerLatchedFailure
  deadline_lane.recovery_close = true
  deadline.multi_prepare_worker_failure_race_terminal(0, OnlineDeadlineReached)
  guard deadline.event_kind is FailureUsageEvent else {
    fail("deadline terminal was rewritten as worker failure")
  }
  assert_true(deadline.events.has_event())
  assert_eq(deadline.multi.event_lane, 0)
  assert_true(deadline_lane.terminal_cut == NoTerminalCut)
  assert_true(deadline_lane.failure_kind == DeadlineLatchedFailure)
  assert_false(deadline_lane.recovery_close)
}

///|
test "multi usage remains request-local and lane reset clears cached usage" {
  let instance = coordinator_fixture_instance(multi_lane_count=2)
  let first = instance.multi.lanes[0]
  let second = instance.multi.lanes[1]
  first.input_tokens = 5
  first.cached_input_tokens = 4
  first.output_tokens = 1
  second.input_tokens = 7
  second.cached_input_tokens = 2
  second.output_tokens = 3

  instance.multi_publish_usage(0, CompletionUsageEvent)
  let first_view = try! instance.events.view()
  let first_usage = try! first_view.usage()
  assert_eq(try! first_usage.input_tokens(), 5)
  assert_eq(try! first_usage.cached_input_tokens(), 4)
  assert_eq(try! first_usage.output_tokens(), 1)
  try! instance.events.retire(first_view)
  instance.multi.event_lane = -1

  instance.multi_publish_usage(1, CompletionUsageEvent)
  let second_view = try! instance.events.view()
  let second_usage = try! second_view.usage()
  assert_eq(try! second_usage.input_tokens(), 7)
  assert_eq(try! second_usage.cached_input_tokens(), 2)
  assert_eq(try! second_usage.output_tokens(), 3)
  try! instance.events.retire(second_view)
  instance.multi.event_lane = -1

  instance.reset_multi_lane(0)
  assert_eq(first.cached_input_tokens, 0)
  assert_eq(second.cached_input_tokens, 2)
}

///|
test "direct publication routes stay one-probe at maximum capacity" {
  let capacity = 65536
  let routes = LunaOnlinePublicationRoutes::new(capacity)
  let first = try! @core.SchedulerPublicationRoute::new(0, 7UL)
  let middle = try! @core.SchedulerPublicationRoute::new(32768, 11UL)
  let last = try! @core.SchedulerPublicationRoute::new(65535, 13UL)
  routes.bind(first, 65535)
  routes.bind(middle, 17)
  routes.bind(last, 0)
  assert_eq(routes.resolve(first), 65535)
  assert_eq(routes.resolve(middle), 17)
  assert_eq(routes.resolve(last), 0)
  assert_eq(routes.lookups, 3UL)
  assert_eq(routes.rejections, 0UL)
  assert_eq(routes.max_probes, 1)
}

///|
test "direct publication routes reject stale generation and exact-clear drift" {
  let routes = LunaOnlinePublicationRoutes::new(8)
  let original = try! @core.SchedulerPublicationRoute::new(6, 41UL)
  let stale = try! @core.SchedulerPublicationRoute::new(6, 40UL)
  let replacement = try! @core.SchedulerPublicationRoute::new(6, 42UL)
  let collision = try! @core.SchedulerPublicationRoute::new(6, 99UL)
  routes.bind(original, 2)
  assert_false(routes.try_bind(collision, 4))
  assert_eq(routes.resolve(stale), -1)
  assert_eq(routes.resolve(original), 2)
  assert_false(routes.try_clear(original, 3))
  routes.clear(original, 2)
  assert_eq(routes.resolve(original), -1)
  routes.bind(replacement, 7)
  assert_eq(routes.resolve(original), -1)
  assert_eq(routes.resolve(replacement), 7)
  assert_eq(routes.lookups, 5UL)
  assert_eq(routes.rejections, 3UL)
  assert_eq(routes.max_probes, 1)
}
