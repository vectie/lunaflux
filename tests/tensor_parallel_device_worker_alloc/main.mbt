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
extern "C" fn fake_readback_count() -> UInt64 = "lunaflux_device_worker_fake_readback_count"

///|
extern "C" fn fake_resource_create_count() -> UInt64 = "lunaflux_device_worker_fake_resource_create_count"

///|
extern "C" fn fake_resource_close_count() -> UInt64 = "lunaflux_device_worker_fake_resource_close_count"

///|
extern "C" fn fake_live_children() -> Int = "lunaflux_device_worker_fake_live_children"

///|
extern "C" fn executor_evidence_reset() = "lunaflux_tp_alloc_evidence_reset"

///|
extern "C" fn executor_enqueues() -> UInt64 = "lunaflux_tp_alloc_executor_enqueues"

///|
extern "C" fn executor_records() -> UInt64 = "lunaflux_tp_alloc_executor_records"

///|
extern "C" fn executor_pending_polls() -> UInt64 = "lunaflux_tp_alloc_executor_pending_polls"

///|
extern "C" fn executor_complete_polls() -> UInt64 = "lunaflux_tp_alloc_executor_complete_polls"

///|
extern "C" fn executor_resets() -> UInt64 = "lunaflux_tp_alloc_executor_resets"

///|
extern "C" fn blocking_synchronizations() -> UInt64 = "lunaflux_tp_alloc_blocking_synchronizations"

///|
extern "C" fn live_executors() -> Int = "lunaflux_tp_alloc_live_executors"

///|
extern "C" fn collective_evidence_reset() = "lunaflux_tp_alloc_collective_evidence_reset"

///|
extern "C" fn collective_submits() -> UInt64 = "lunaflux_tp_alloc_collective_submits"

///|
extern "C" fn collective_pending_polls() -> UInt64 = "lunaflux_tp_alloc_collective_pending_polls"

///|
extern "C" fn collective_complete_polls() -> UInt64 = "lunaflux_tp_alloc_collective_complete_polls"

///|
extern "C" fn live_communicators() -> Int = "lunaflux_tp_alloc_live_communicators"

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
  let observed = alloc_probe_end()
  positive_control_sink[0] = None
  if alloc_probe_check(observed, 0) != 1 || alloc_probe_direct_count() == 0UL {
    alloc_probe_abort()
  }
}

///|
fn positive_array_control() -> Unit {
  alloc_probe_begin()
  let array = FixedArray::make(8, alloc_probe_seed())
  keep_array_alive(array)
  let observed = alloc_probe_end()
  if alloc_probe_check(observed, 0) != 1 || alloc_probe_array_count() == 0UL {
    alloc_probe_abort()
  }
}

///|
fn positive_string_control() -> Unit {
  alloc_probe_begin()
  let dynamic = "tp-allocation-control-" + alloc_probe_seed().to_string()
  string_control_sink[0] = Some(dynamic)
  let observed = alloc_probe_end()
  string_control_sink[0] = None
  if alloc_probe_check(observed, 0) != 1 || alloc_probe_string_count() == 0UL {
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
fn execute_rank_cycle(
  workers : WorkerFixture,
  cycles : CycleFixture,
  index : Int,
) -> Unit raise {
  for rank in 0..<TP_WORLD_SIZE {
    execute_cycle(workers.workers[rank], rank, cycles.cycles[index])
  }
}

///|
fn verify_warmed_evidence(
  observed : UInt64,
  measured_cycle_count : Int,
  workers : WorkerFixture,
  warm_copies : UInt64,
  warm_launches : UInt64,
  warm_reads : UInt64,
  copies_before : UInt64,
  launches_before : UInt64,
  reads_before : UInt64,
  creates_before : UInt64,
  closes_before : UInt64,
) -> Unit {
  let rank_cycles = measured_cycle_count.to_uint64() * TP_WORLD_SIZE.to_uint64()
  let expected_enqueues = rank_cycles * workers.operation_count.to_uint64()
  let expected_collectives = rank_cycles * workers.collective_count.to_uint64()
  if alloc_probe_check(observed, 1) != 1 ||
    executor_enqueues() != expected_enqueues ||
    executor_records() != rank_cycles ||
    executor_pending_polls() != rank_cycles ||
    executor_complete_polls() != rank_cycles ||
    executor_resets() != rank_cycles ||
    collective_submits() != expected_collectives ||
    collective_pending_polls() != expected_collectives ||
    collective_complete_polls() != expected_collectives ||
    blocking_synchronizations() != 0UL ||
    fake_copy_count() - copies_before !=
    warm_copies * measured_cycle_count.to_uint64() ||
    fake_launch_count() - launches_before !=
    warm_launches * measured_cycle_count.to_uint64() ||
    fake_readback_count() - reads_before !=
    warm_reads * measured_cycle_count.to_uint64() ||
    fake_resource_create_count() != creates_before ||
    fake_resource_close_count() != closes_before {
    alloc_probe_abort()
  }
}

///|
async fn run_allocation_evidence() -> Unit {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-tp-device-worker-alloc")
    let directory = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let weights = fixture_weight_file()
    @fs.write_file(
      "\{directory}/weights.safetensors",
      weights,
      create_mode=CreateNew,
    )
    let root = @approved_fs.ApprovedRoot::open_absolute(directory)
    let execution = prepare_execution_fixture(root, weights)
    let workers = prepare_workers(root, execution)
    let measured_cycle_count = 65
    let cycles = prepare_cycles(measured_cycle_count + 1, execution.model)
    verify_probe_controls()
    let copies_before_warm = fake_copy_count()
    let launches_before_warm = fake_launch_count()
    let reads_before_warm = fake_readback_count()
    execute_rank_cycle(workers, cycles, 0)
    let warm_copies = fake_copy_count() - copies_before_warm
    let warm_launches = fake_launch_count() - launches_before_warm
    let warm_reads = fake_readback_count() - reads_before_warm
    if warm_copies == 0UL ||
      warm_launches != workers.operation_count.to_uint64() * 2UL ||
      warm_reads != 1UL {
      alloc_probe_abort()
    }
    let copies_before = fake_copy_count()
    let launches_before = fake_launch_count()
    let reads_before = fake_readback_count()
    let creates_before = fake_resource_create_count()
    let closes_before = fake_resource_close_count()
    executor_evidence_reset()
    collective_evidence_reset()
    alloc_probe_begin()
    for index in 1..<=measured_cycle_count {
      execute_rank_cycle(workers, cycles, index)
    }
    let observed = alloc_probe_end()
    verify_warmed_evidence(
      observed, measured_cycle_count, workers, warm_copies, warm_launches, warm_reads,
      copies_before, launches_before, reads_before, creates_before, closes_before,
    )
    release_cycles(cycles)
    close_workers(workers)
    root.close()
    if fake_live_children() != 0 ||
      live_executors() != 0 ||
      live_communicators() != 0 {
      alloc_probe_abort()
    }
  }
}

///|
async fn main {
  run_allocation_evidence()
}
