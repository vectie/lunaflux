///|
extern "C" fn alloc_probe_begin() = "lunaflux_device_worker_alloc_probe_begin"

///|
extern "C" fn alloc_probe_end() -> UInt64 = "lunaflux_device_worker_alloc_probe_end"

///|
extern "C" fn alloc_probe_direct_count() -> UInt64 = "lunaflux_device_worker_alloc_probe_direct_count"

///|
extern "C" fn alloc_probe_array_count() -> UInt64 = "lunaflux_device_worker_alloc_probe_array_count"

///|
extern "C" fn alloc_probe_string_count() -> UInt64 = "lunaflux_device_worker_alloc_probe_string_count"

///|
extern "C" fn alloc_probe_check(observed : UInt64, expect_zero : Int) -> Int = "lunaflux_device_worker_alloc_probe_check"

///|
extern "C" fn alloc_probe_abort() = "lunaflux_device_worker_alloc_probe_abort"

///|
extern "C" fn alloc_probe_seed() -> Int = "lunaflux_device_worker_alloc_probe_seed"

///|
#borrow(value)
extern "C" fn keep_array_alive(value : FixedArray[Int]) = "lunaflux_device_worker_alloc_probe_keep_alive"

///|
extern "C" fn fake_copy_count() -> UInt64 = "lunaflux_device_worker_fake_copy_count"

///|
extern "C" fn fake_launch_count() -> UInt64 = "lunaflux_device_worker_fake_launch_count"

///|
extern "C" fn fake_sync_count() -> UInt64 = "lunaflux_device_worker_fake_sync_count"

///|
extern "C" fn fake_readback_count() -> UInt64 = "lunaflux_device_worker_fake_readback_count"

///|
extern "C" fn fake_live_children() -> Int = "lunaflux_device_worker_fake_live_children"

///|
extern "C" fn fake_resource_create_count() -> UInt64 = "lunaflux_device_worker_fake_resource_create_count"

///|
extern "C" fn fake_resource_close_count() -> UInt64 = "lunaflux_device_worker_fake_resource_close_count"

///|
struct PositiveControl {
  _value : Int
}

///|
let positive_control_sink : FixedArray[PositiveControl?] = FixedArray::make(
  1,
  None,
)

///|
let string_control_sink : FixedArray[String?] = FixedArray::make(1, None)

///|
fn positive_record_control() -> Unit {
  alloc_probe_begin()
  let record = { _value: alloc_probe_seed() }
  positive_control_sink[0] = Some(record)
  let record_observed = alloc_probe_end()
  positive_control_sink[0] = None
  if alloc_probe_check(record_observed, 0) != 1 ||
    alloc_probe_direct_count() == 0UL {
    alloc_probe_abort()
  }
}

///|
fn positive_array_control() -> Unit {
  alloc_probe_begin()
  let array = FixedArray::make(8, alloc_probe_seed())
  keep_array_alive(array)
  let array_observed = alloc_probe_end()
  if alloc_probe_check(array_observed, 0) != 1 ||
    alloc_probe_array_count() == 0UL {
    alloc_probe_abort()
  }
}

///|
fn positive_string_control() -> Unit {
  alloc_probe_begin()
  let dynamic = "allocation-control-" + alloc_probe_seed().to_string()
  string_control_sink[0] = Some(dynamic)
  let string_observed = alloc_probe_end()
  string_control_sink[0] = None
  if alloc_probe_check(string_observed, 0) != 1 ||
    alloc_probe_string_count() == 0UL {
    alloc_probe_abort()
  }
}

///|
fn verify_probe_controls() -> Unit {
  positive_record_control()
  positive_array_control()
  positive_string_control()
}

///|
struct PreparedCycle {
  plan_owner : @worker_protocol.SchedulePlanBuffer
  plan : @worker_protocol.SubmittedSchedulePlan
  frame : @worker_wire.ValidatedPlanFrame
}

///|
fn prepare_cycle(
  sequence : UInt64,
  model_plan : @plan.ModelPlan,
  page : @page_allocator.PageId,
) -> PreparedCycle raise {
  let limits = fixture_worker_limits()
  let plan_owner = @worker_protocol.SchedulePlanBuffer::new(limits)
  plan_owner.append_final_prefill(
    request_id=@inference.RequestId::new(sequence),
    request_generation=@inference.RequestGeneration::initial(),
    sequence_token_start=0,
    tokens=[0],
    pages=[page],
    capabilities=model_plan.required_capabilities(),
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(sequence),
    sample_index=0,
    completion_slot=@worker_protocol.CompletionSlot::new(0, limits),
  )
  let generation = @worker_protocol.ModelPlanGeneration::first()
  let plan = plan_owner.submit(
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
  let frame_owner = @worker_wire.PlanFrameBuffer::new(
    limits,
    fixture_inference_limits(),
  )
  { plan_owner, plan, frame: frame_owner.encode_plan(plan) }
}

///|
fn prepare_cycles(
  count : Int,
  model_plan : @plan.ModelPlan,
) -> FixedArray[PreparedCycle] raise {
  let allocator = @page_allocator.PageAllocator::new(
    @page_allocator.PageAllocatorLimits::new(
      page_capacity=1,
      max_run_pages=1,
      max_active_references_per_page=1,
      max_cached_references_per_page=1,
    ),
  )
  let page = allocator.allocate_one()
  let first = prepare_cycle(1UL, model_plan, page)
  let cycles = FixedArray::make(count, first)
  for index in 1..<count {
    cycles[index] = prepare_cycle((index + 1).to_uint64(), model_plan, page)
  }
  cycles
}

///|
fn execute_cycle(
  owner : @device_worker.DeviceWorkerOwner,
  completion_owner : @worker_wire.CompletionFrameBuffer,
  cycle : PreparedCycle,
) -> Unit raise {
  let writer = completion_owner.writer_for(cycle.frame)
  let completion = owner.execute_frame(cycle.frame, writer)
  completion.validate_for(cycle.plan)
  if completion.sequence_value() != cycle.plan.sequence_wire() ||
    completion.model_generation_value() != 1UL ||
    completion.entry_count() != 1 {
    alloc_probe_abort()
  }
  let entry = completion.entry(0)
  if entry.kind_value() != 1 ||
    entry.request_id_value() != cycle.plan.sequence_wire() ||
    entry.request_generation_value() != 0UL ||
    entry.processed_tokens() != 1 ||
    entry.token_id() != 0 {
    alloc_probe_abort()
  }
  let retired = cycle.plan_owner.retire(cycle.plan)
  cycle.plan_owner.reset(retired)
}

///|
async fn run_allocation_evidence() -> Unit {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-device-worker-alloc")
    let directory = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let weights = fixture_weight_file()
    let module_bytes = b"\x01\x02\x03"
    let model = fixture_model(weights)
    let model_plan = @llama.build_paged(model)
    let target = @catalog.DeviceTarget::new(
      compute_major=8,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
    )
    let manifest = fixture_manifest(model_plan, target, module_bytes)
    write_fixture_files(directory, weights, manifest, module_bytes)
    let owner = prepare_fixture_owner(
      directory, manifest, model, model_plan, target,
    )
    let cycle_count = 128
    let cycles = prepare_cycles(cycle_count + 1, model_plan)
    let completion_owner = @worker_wire.CompletionFrameBuffer::new(
      fixture_worker_limits(),
      fixture_inference_limits(),
    )
    verify_probe_controls()
    execute_cycle(owner, completion_owner, cycles[0])
    let copies_before = fake_copy_count()
    let launches_before = fake_launch_count()
    let syncs_before = fake_sync_count()
    let reads_before = fake_readback_count()
    let creates_before = fake_resource_create_count()
    let closes_before = fake_resource_close_count()
    alloc_probe_begin()
    for index in 1..<=cycle_count {
      execute_cycle(owner, completion_owner, cycles[index])
    }
    let observed = alloc_probe_end()
    let operation_count = model_plan.operations().length().to_uint64()
    if alloc_probe_check(observed, 1) != 1 ||
      fake_copy_count() - copies_before != cycle_count.to_uint64() * 8UL ||
      fake_launch_count() - launches_before !=
      cycle_count.to_uint64() * operation_count ||
      fake_sync_count() - syncs_before !=
      cycle_count.to_uint64() * operation_count ||
      fake_readback_count() - reads_before != cycle_count.to_uint64() ||
      fake_resource_create_count() != creates_before ||
      fake_resource_close_count() != closes_before {
      alloc_probe_abort()
    }
    owner.close()
    if fake_live_children() != 0 {
      alloc_probe_abort()
    }
    run_failure_evidence(directory, manifest, model, model_plan, target)
  }
}

///|
async fn main {
  run_allocation_evidence()
}
