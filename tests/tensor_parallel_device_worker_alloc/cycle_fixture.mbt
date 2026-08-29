///|
priv struct PreparedCycle {
  frame : @worker_wire.ValidatedPlanFrame
  sequence : UInt64
}

///|
priv struct CycleFixture {
  cycles : FixedArray[PreparedCycle]
  allocator : @page_allocator.PageAllocator
  page : @page_allocator.PageId
}

///|
fn prepare_cycle(
  sequence : UInt64,
  model : @plan.ModelPlan,
  page : @page_allocator.PageId,
) -> PreparedCycle raise {
  let limits = fixture_worker_limits()
  let owner = @worker_protocol.SchedulePlanBuffer::new(limits)
  owner.append_final_prefill(
    request_id=@inference.RequestId::new(sequence + 100UL),
    request_generation=@inference.RequestGeneration::from_wire(sequence),
    sequence_token_start=0,
    tokens=[(sequence % 8UL).to_int()],
    pages=[page],
    capabilities=model.required_capabilities(),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(sequence + 500UL),
    sample_index=0,
    completion_slot=@worker_protocol.CompletionSlot::new(0, limits),
  )
  let generation = @worker_protocol.ModelPlanGeneration::from_wire(
    TP_MODEL_GENERATION,
  )
  let plan = owner.submit(
    sequence=@worker_protocol.PlanSequence::from_wire(sequence),
    model_plan_generation=generation,
    token_budget=1,
    if sequence == 1UL {
      None
    } else {
      Some(@worker_protocol.PlanSequence::from_wire(sequence - 1UL))
    },
    generation,
  )
  let frame = @worker_wire.PlanFrameBuffer::new(
    limits,
    fixture_inference_limits(),
  ).encode_plan(plan)
  { frame, sequence }
}

///|
fn prepare_cycles(count : Int, model : @plan.ModelPlan) -> CycleFixture raise {
  if count < 2 {
    abort("TP allocation fixture requires warm and measured cycles")
  }
  let allocator = @page_allocator.PageAllocator::new(
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=1,
      max_run_pages=1,
      max_active_references_per_page=1,
      max_cached_references_per_page=1,
    ),
  )
  let page = allocator.allocate_one()
  let first = prepare_cycle(1UL, model, page)
  let cycles = FixedArray::make(count, first)
  for index in 1..<count {
    cycles[index] = prepare_cycle((index + 1).to_uint64(), model, page)
  }
  { cycles, allocator, page }
}

///|
fn finish_progress(
  worker : @tensor_parallel_device_worker.TensorParallelDeviceWorkerOwner,
  staged : @tensor_parallel_device_worker.StagedTensorParallelDeviceExecution,
  frame : @worker_wire.ValidatedPlanFrame,
) -> @tensor_parallel_device_worker.ExecutedTensorParallelDeviceExecution raise {
  let mut attempts = 0
  for ;; {
    let progress = worker.progress_execution(staged, frame)
    attempts += 1
    if !progress.is_pending() {
      if attempts < 3 || attempts > 128 {
        alloc_probe_abort()
      }
      return progress.executed()
    }
    if attempts > 128 {
      alloc_probe_abort()
    }
  }
}

///|
fn execute_cycle(
  worker : @tensor_parallel_device_worker.TensorParallelDeviceWorkerOwner,
  rank : Int,
  cycle : PreparedCycle,
) -> Unit raise {
  let staged = worker.stage_frame(cycle.frame)
  let executed = finish_progress(worker, staged, cycle.frame)
  if rank == 0 {
    let completion = worker.finish_leader_frame(executed, cycle.frame)
    if completion.sequence_value() != cycle.sequence ||
      completion.model_generation_value() != TP_MODEL_GENERATION ||
      completion.entry_count() != 1 ||
      completion.entry(0).token_id() != 0 {
      alloc_probe_abort()
    }
  } else {
    let ack = worker.finish_follower(executed, cycle.frame)
    if ack.group_generation_value() != TP_GROUP_GENERATION ||
      ack.model_generation_value() != TP_MODEL_GENERATION ||
      ack.plan_sequence_value() != cycle.sequence ||
      ack.rank() != rank ||
      ack.world_size() != TP_WORLD_SIZE {
      alloc_probe_abort()
    }
  }
  if worker.lifecycle() != WorkerReady {
    alloc_probe_abort()
  }
  let report = worker.graph_runtime_telemetry(cycle.sequence)
  if report.sequence_value() != cycle.sequence ||
    report.selected_path() != WorkerGraphOrderedEager ||
    report.has_selected_shape_class() ||
    report.graph_hits() != 0UL ||
    report.graph_misses() != cycle.sequence {
    alloc_probe_abort()
  }
}

///|
fn release_cycles(fixture : CycleFixture) -> Unit raise {
  fixture.allocator.release_active(fixture.page)
  if fixture.allocator.allocated_page_count() != 0 ||
    fixture.allocator.free_page_count() != 1 {
    alloc_probe_abort()
  }
}
