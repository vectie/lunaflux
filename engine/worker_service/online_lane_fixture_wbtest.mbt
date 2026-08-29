///|
fn wb_online_lane_lease(
  service : WorkerService,
  clock : @monotonic_clock.MonotonicClock,
  prepared_admission : @core.PreparedExclusiveAdmission?,
  prepared_clock_read : @monotonic_clock.PreparedMonotonicRead,
  handle : @core.RequestHandle?,
  phase : Int,
  request_generation : @inference.RequestGeneration,
  cut_source_generation : @inference.RequestGeneration,
  restart_disposition : Int,
  expected_position : Int,
  cancel_reserved : Bool,
  semantic_retention : @inference.LunaRequestStopTokenRetentionSlot,
  epoch? : UInt64 = 1UL,
  publication_cursor? : UInt64 = 0UL,
  lane_count? : Int = 1,
) -> OnlineWorkerLease {
  let lane_handles = FixedArray::makei(lane_count, _ => Array::new(capacity=1))
  match handle {
    Some(value) => lane_handles[0].push(value)
    None => ()
  }
  let occupied = phase != ONLINE_LANE_EMPTY
  {
    service,
    epoch,
    clock,
    restart_policy: try! WorkerRestartBackoffPolicy::new(
      initial_delay_millis=1UL,
      maximum_delay_millis=8UL,
      stable_success_millis=4UL,
      maximum_attempts=8,
    ),
    restart_attempts: 0,
    restart_generation: 0UL,
    restart_phase: 0,
    restart_delay_millis: 0UL,
    restart_deadline_millis: 0UL,
    restart_cached_remaining_millis: 0,
    restart_stability_generation: 0UL,
    restart_stability_deadline_millis: 0UL,
    restart_stability_predecessor: 0UL,
    restart_clock_initialized: false,
    restart_last_clock_millis: 0UL,
    restart_forbidden: false,
    prepared_admission,
    prepared_clock_read,
    lane_phases: FixedArray::makei(lane_count, index => {
      if index == 0 {
        phase
      } else {
        ONLINE_LANE_EMPTY
      }
    }),
    lane_generations: FixedArray::makei(lane_count, index => {
      if index == 0 && occupied {
        1UL
      } else {
        0UL
      }
    }),
    lane_handles,
    lane_semantic_retentions: FixedArray::makei(lane_count, index => {
      if index == 0 {
        semantic_retention
      } else {
        @inference.LunaRequestStopTokenRetentionSlot::new()
      }
    }),
    lane_request_generations: FixedArray::makei(lane_count, index => {
      if index == 0 {
        request_generation
      } else {
        @inference.RequestGeneration::initial()
      }
    }),
    lane_cut_source_generations: FixedArray::makei(lane_count, index => {
      if index == 0 {
        cut_source_generation
      } else {
        @inference.RequestGeneration::initial()
      }
    }),
    lane_restart_dispositions: FixedArray::makei(lane_count, index => {
      if index == 0 {
        restart_disposition
      } else {
        0
      }
    }),
    lane_expected_positions: FixedArray::makei(lane_count, index => {
      if index == 0 {
        expected_position
      } else {
        0
      }
    }),
    free_lanes: FixedArray::makei(lane_count, index => index),
    free_head: if occupied {
      1 % lane_count
    } else {
      0
    },
    free_count: if occupied {
      lane_count - 1
    } else {
      lane_count
    },
    active_lane_count: if occupied {
      1
    } else {
      0
    },
    compatibility_lane: if occupied {
      0
    } else {
      -1
    },
    compatibility_generation: if occupied {
      1UL
    } else {
      0UL
    },
    publication_cursor,
    cancel_lane: if cancel_reserved {
      0
    } else {
      -1
    },
    cancel_generation: if cancel_reserved {
      1UL
    } else {
      0UL
    },
  }
}

///|
fn wb_admit_routed(
  scheduler : @core.Scheduler,
  request : @core.TokenizedRequest,
  received_at? : UInt64 = 0UL,
) -> @core.RequestHandle {
  let prepared = scheduler.prepare_admission(request)
  try! prepared.bind_publication_route(
    try! @core.SchedulerPublicationRoute::new(0, 1UL),
  )
  try! scheduler.commit_admission(prepared, received_at)
  try! prepared.request_handle()
}
