///|
test "paged profiler counters are bounded sorted observations only" {
  let observation = @luna_profile_priority.LunaPagedKernelProfileObservation::new(
    operation_id=@plan.OperationId::new(4),
    shape=paged_profile_shape(),
    call_count=10L,
    self_time_ns=600L,
    inclusive_time_ns=700L,
    counters=[
      @luna_profile_priority.LunaPagedProfileCounter::new(
        kind=KernelLaunchCount,
        value=10L,
      ),
      @luna_profile_priority.LunaPagedProfileCounter::new(
        kind=DramReadBytes,
        value=4096L,
      ),
    ],
  )
  assert_eq(observation.counters().length(), 2)
  assert_eq(observation.counters()[1].value(), 4096L)
  try
    @luna_profile_priority.LunaPagedKernelProfileObservation::new(
      operation_id=@plan.OperationId::new(4),
      shape=paged_profile_shape(),
      call_count=10L,
      self_time_ns=600L,
      inclusive_time_ns=700L,
      counters=[
        @luna_profile_priority.LunaPagedProfileCounter::new(
          kind=DramReadBytes,
          value=1L,
        ),
        @luna_profile_priority.LunaPagedProfileCounter::new(
          kind=KernelLaunchCount,
          value=1L,
        ),
      ],
    )
  catch {
    @luna_profile_priority.LunaKernelProfileInvalid(CounterOrder) => ()
    _ => fail("counter-order rejection returned the wrong rule")
  } noraise {
    _ => fail("unsorted counters were admitted")
  }
}
