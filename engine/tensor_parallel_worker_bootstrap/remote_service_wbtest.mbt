///|
// Protocol emulator: synthetic token values test the real scheduler/service
// composition; this is not GPU numerical qualification.
struct ServiceRankEmulator {
  mailbox : @remote_mailbox.RemoteMailbox
  control : @rank_child_control.RankChildControl
  plan : @worker_wire.PlanFrameBuffer
  completion : @worker_wire.CompletionFrameBuffer
  bytes : FixedArray[Byte]
  response : FixedArray[Byte]
  mut response_length : Int
  mut sequence : UInt64
  mut phase : Int
}

///|
fn ServiceRankEmulator::new(
  startup : @worker_wire.WorkerStartupContract,
  limits : @rank_group_wire.RankGroupWireLimits,
) -> ServiceRankEmulator raise {
  let mailbox = @remote_mailbox.RemoteMailbox::new(limits.max_frame_bytes())
  let control = @rank_child_control.open_remote_rank_child_control(
    mailbox, limits,
  )
  control.begin_receive()
  {
    mailbox,
    control,
    plan: @worker_wire.PlanFrameBuffer::new(
      startup.worker_limits(),
      startup.inference_limits(),
    ),
    completion: @worker_wire.CompletionFrameBuffer::new(
      startup.worker_limits(),
      startup.inference_limits(),
    ),
    bytes: FixedArray::make(limits.max_frame_bytes(), b'\x00'),
    response: FixedArray::make(limits.max_payload_bytes(), b'\x00'),
    response_length: 0,
    sequence: 0UL,
    phase: 0,
  }
}

///|
fn ServiceRankEmulator::progress(
  self : ServiceRankEmulator,
  group : @remote_group.RemoteGroup,
  rank : Int,
) -> Unit raise {
  bridge_remote_mailbox(group.mailbox(rank), self.mailbox, self.bytes)
  if self.phase == 0 && self.control.progress_receive().has_inbound() {
    let inbound = self.control.inbound()
    match inbound.kind() {
      Configure => {
        self.control.begin_ready(
          inbound,
          inbound.binding(),
          previous_sequence_value=0UL,
        )
        self.phase = 1
      }
      Submit => {
        inbound.copy_payload_to(self.bytes, destination_offset=0)
        let frame = self.plan.load(self.bytes, inbound.payload_length())
        self.sequence = frame.sequence_value()
        let writer = self.completion.writer_for(frame)
        for i in 0..<frame.prefill_count() {
          writer.append_final_prefill_completed(frame.prefill_row(i), 3)
        }
        for i in 0..<frame.decode_count() {
          writer.append_decode_completed(frame.decode_row(i), 4)
        }
        let result = writer.submit()
        result.copy_to(self.response, destination_offset=0)
        self.response_length = result.length()
        self.control.accept_submit(inbound)
        self.control.begin_receive()
      }
      PollComplete => {
        if rank == 0 {
          self.control.begin_leader_complete(
            inbound,
            self.response,
            payload_length=self.response_length,
          )
        } else {
          self.control.begin_follower_complete(inbound)
        }
        self.phase = 2
      }
      Drain => {
        self.control.begin_drained(inbound)
        self.phase = 1
      }
      Close => {
        self.control.begin_closed(inbound)
        self.phase = 3
      }
      _ => fail("unexpected remote service command")
    }
  }
  bridge_remote_mailbox(self.mailbox, group.mailbox(rank), self.bytes)
  if self.phase > 0 &&
    self.phase < 4 &&
    self.control.progress_send().is_complete() {
    if self.phase == 2 {
      self.control.begin_graph_telemetry_sidecar(
        @worker_wire.WorkerGraphTelemetry::without_shape(
          sequence=self.sequence,
          path=WorkerGraphOrderedEager,
          graph_hits=0UL,
          graph_misses=self.sequence,
        ),
      )
      self.phase = 1
    } else if self.phase == 3 {
      self.phase = 4
    } else {
      self.phase = 0
      self.control.begin_receive()
    }
  }
}

///|
fn remote_test_service_preparation(
  remote : @remote_startup.RemoteGroupStartup,
  admitted : TensorParallelGroupBootstrap,
  model : @plan.ModelPlan,
) -> @worker_service.RemoteWorkerServicePreparation raise {
  let startup = remote.rank(0).unwrap().worker()
  let worker = startup.worker_limits()
  let limits = @rank_group_wire.RankGroupWireLimits::new(
    max_payload_bytes=16384,
  )
  let scheduling = @scheduler_config.new(
    step_token_budget=16,
    max_active_requests=2,
    max_waiting_requests=2,
    prefill_chunk_tokens=8,
    emergency_decode_page_reserve=1,
    waiting_age_threshold_steps=2,
    output_event_capacity=4,
  )
  let cache = @cache.new(
    tokens_per_page=128,
    total_page_count=4,
    block_table_pages_per_request=1,
    prefix_enabled=false,
    max_prefix_entries=0,
    max_prefix_nodes=0,
    max_prefix_tokens_per_entry=0,
    max_prefix_pages=0,
    max_active_references_per_page=4,
    max_cached_references_per_page=4,
    layout_version=1,
  )
  let capacity = @runtime_resolved.resolve(
    scheduling,
    cache,
    model.shape_constraints(),
    worker,
  )
  let generation = @worker_protocol.ModelPlanGeneration::from_wire(
    startup.model_generation_value(),
  )
  let recipe = @worker_protocol.RowCapabilityRecipe::new(
    identity=model.identity(),
    model_plan_generation=generation,
    [@plan.KernelCapabilityId::embedding_lookup()],
    worker,
  )
  let blueprint = @core.SchedulerBlueprint::new(
    capacity,
    model.identity(),
    generation,
    recipe,
    recipe,
    recipe,
  )
  @worker_service.prepare_owned_remote(
    blueprint,
    remote,
    admitted,
    limits,
    timeout_millis=10000UL,
    @worker_service.WorkerRestartBackoffPolicy::new(
      initial_delay_millis=1UL,
      maximum_delay_millis=10UL,
      stable_success_millis=100UL,
      maximum_attempts=2,
    ),
  )
}

///|
fn exercise_remote_service(
  remote : @remote_startup.RemoteGroupStartup,
  admitted : TensorParallelGroupBootstrap,
  model : @plan.ModelPlan,
) -> Unit raise {
  let startup = remote.rank(0).unwrap().worker()
  let limits = @rank_group_wire.RankGroupWireLimits::new(
    max_payload_bytes=16384,
  )
  let pending = remote_test_service_preparation(remote, admitted, model)
  let group = pending.control().group()
  let ranks = [0, 1].map(_ => ServiceRankEmulator::new(startup, limits))
  let service = for _ in 0..<100 {
    for rank in 0..<2 {
      ranks[rank].progress(group, rank)
    }
    if pending.poll_ready() is Some(ready) {
      break ready.take_raw_ready()
    }
  } nobreak {
    fail("remote service never ready")
  }
  let inference = startup.inference_limits()
  let storage = @inference.LunaRequestSemanticStorage::new(
    inference,
    @inference.LunaRequestSemanticStepBudget::new(17),
  )
  let write = storage.begin(
    stop_token_count=0,
    stop_string_count=0,
    cache_scope_byte_count=1,
    cache_permission=Disabled,
  )
  write.push_cache_scope_byte(b'x')
  let work = write.finish()
  while work.progress().is_pending() {

  }
  let semantic = work.take_lease()
  defer (semantic.release() catch { _ => () })
  let request = @core.TokenizedRequest::new(
    request_id=@inference.RequestId::new(99UL),
    model_identity=model.identity(),
    input_tokens=@inference.TokenBuffer::new([1], inference),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(99UL),
    stop_tokens=semantic.stop_token_view(),
    stream_preference=Streaming,
    effective_limits=@inference.EffectiveLimits::new(
      max_new_tokens=2,
      context_ceiling=16,
      inference,
    ),
    admission_deadline=@inference.AdmissionDeadline::from_receipt(
      0UL,
      @inference.DeadlineBudget::from_milliseconds(1000UL, inference),
    ),
  )
  ignore(service.admit(request, 0UL))
  for _ in 0..<1000 {
    for rank in 0..<2 {
      ranks[rank].progress(group, rank)
    }
    ignore(service.progress())
    if service.live_request_count() == 0 && service.outstanding_count() == 0 {
      break
    }
  }
  assert_eq(service.live_request_count(), 0)
  assert_eq(service.outstanding_count(), 0)
  assert_eq(service.generated_token_count(), 2)
  assert_eq(service.take_generated_token().token_id(), 3)
  assert_eq(service.take_generated_token().token_id(), 4)
  assert_eq(service.terminal_notice_count(), 1)
  ignore(service.take_terminal_notice())
  service.begin_shutdown_maintenance()
  for _ in 0..<100 {
    for rank in 0..<2 {
      ranks[rank].progress(group, rank)
    }
    ignore(service.progress_maintenance())
    if service.lifecycle() == @worker_service.ServiceClosed {
      break
    }
  }
  assert_true(service.lifecycle() == @worker_service.ServiceClosed)
  assert_true(group.lifecycle() == @remote_group.Closed)
}
