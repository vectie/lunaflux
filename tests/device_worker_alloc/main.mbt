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
    let cycle_count = 131
    let matrix = prepare_cycles(cycle_count + 1, model_plan)
    let cycles = matrix.cycles
    let completion_owner = @worker_wire.CompletionFrameBuffer::new(
      fixture_worker_limits(),
      fixture_inference_limits(),
    )
    let operation_count = model_plan.operations().length().to_uint64()
    verify_probe_controls()
    execute_cycle(owner, completion_owner, cycles[0], operation_count)
    let copies_before = fake_copy_count()
    let launches_before = fake_launch_count()
    let syncs_before = fake_sync_count()
    let reads_before = fake_readback_count()
    let creates_before = fake_resource_create_count()
    let closes_before = fake_resource_close_count()
    alloc_probe_begin()
    let mut ordinary_prefills = 0
    let mut greedy_final_prefills = 0
    let mut stochastic_final_prefills = 0
    let mut greedy_decodes = 0
    let mut stochastic_decodes = 0
    for index in 1..<=cycle_count {
      let cycle = cycles[index]
      execute_cycle(owner, completion_owner, cycle, operation_count)
      if cycle.branch == CYCLE_ORDINARY_PREFILL {
        ordinary_prefills += 1
      } else if cycle.branch == CYCLE_GREEDY_FINAL_PREFILL {
        greedy_final_prefills += 1
      } else if cycle.branch == CYCLE_STOCHASTIC_FINAL_PREFILL {
        stochastic_final_prefills += 1
      } else if cycle.branch == CYCLE_GREEDY_DECODE {
        greedy_decodes += 1
      } else if cycle.branch == CYCLE_STOCHASTIC_DECODE {
        stochastic_decodes += 1
      } else {
        alloc_probe_abort()
      }
    }
    let observed = alloc_probe_end()
    if alloc_probe_check(observed, 1) != 1 ||
      ordinary_prefills != 43 ||
      greedy_final_prefills != 22 ||
      stochastic_final_prefills != 22 ||
      greedy_decodes != 22 ||
      stochastic_decodes != 22 ||
      fake_copy_count() - copies_before != cycle_count.to_uint64() * 8UL ||
      fake_launch_count() - launches_before !=
      cycle_count.to_uint64() * operation_count ||
      fake_sync_count() - syncs_before !=
      cycle_count.to_uint64() * operation_count ||
      fake_readback_count() - reads_before != 88UL ||
      fake_resource_create_count() != creates_before ||
      fake_resource_close_count() != closes_before {
      alloc_probe_abort()
    }
    matrix.release_pages()
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
