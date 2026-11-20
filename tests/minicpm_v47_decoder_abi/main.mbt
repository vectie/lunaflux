///|
extern "C" fn live_resources() -> Int32 = "lunaflux_device_worker_fake_live_children"

///|
extern "C" fn launches() -> UInt64 = "lunaflux_device_worker_fake_launch_count"

///|
extern "C" fn creates() -> UInt64 = "lunaflux_device_worker_fake_resource_create_count"

///|
extern "C" fn closes() -> UInt64 = "lunaflux_device_worker_fake_resource_close_count"

///|
extern "C" fn copies() -> UInt64 = "lunaflux_device_worker_fake_copy_count"

///|
extern "C" fn reads() -> UInt64 = "lunaflux_device_worker_fake_readback_count"

///|
extern "C" fn set_sample_token(token : Int32) = "lunaflux_device_worker_fake_set_sample_token"

///|
extern "C" fn set_fault(mode : Int32) = "lunaflux_device_worker_fake_set_fault"

///|
extern "C" fn alloc_begin() = "lunaflux_device_worker_alloc_probe_begin"

///|
extern "C" fn alloc_end() -> UInt64 = "lunaflux_device_worker_alloc_probe_end"

///|
extern "C" fn seed() -> Int32 = "lunaflux_device_worker_alloc_probe_seed"

///|
#borrow(value)
extern "C" fn keep_alive(value : FixedArray[Int32]) = "lunaflux_device_worker_alloc_probe_keep_alive"

///|
fn submit(executor : @device.OrderedKernelExecutor) -> Unit raise {
  for index in 0..<executor.kernel_count() {
    executor.enqueue(index)
  }
  executor.record_completion()
  executor.wait_completion()
}

///|
fn measured_submit(
  executor : @device.OrderedKernelExecutor,
  reset~ : Bool = false,
) -> Unit raise {
  let before = (creates(), closes(), copies(), reads())
  alloc_begin()
  if reset {
    executor.reset()
  }
  submit(executor)
  let observed = alloc_end()
  assert_eq(observed, 0UL)
  assert_eq((creates(), closes(), copies(), reads()), before)
}

///|
fn measured_request_step(
  runner : @request.Runner,
  produces_token~ : Bool,
  poll~ : Bool = false,
) -> Unit raise {
  let before_create = creates()
  let before_close = closes()
  let before_copy = copies()
  let before_read = reads()
  alloc_begin()
  let submitted = runner.submit_next()
  let completed = if poll {
    runner.poll()
  } else {
    runner.wait()
    true
  }
  let observed = alloc_end()
  assert_true(submitted && completed)
  assert_eq(observed, 0UL)
  assert_eq(creates(), before_create)
  assert_eq(closes(), before_close)
  assert_eq(copies() - before_copy, 10UL)
  assert_eq(reads() - before_read, if produces_token { 1UL } else { 0UL })
}

///|
fn upload_i32(
  context : @device.Context,
  allocation : @device.Allocation,
  values : ArrayView[Int32],
) -> Unit raise {
  let bytes = FixedArray::makei(values.length() * 4, index => {
    (values[index / 4].reinterpret_as_uint() >> (index % 4 * 8)).to_byte()
  })
  context.copy_from_fixed_host(
    allocation,
    bytes,
    0L,
    0L,
    bytes.length().to_int64(),
  )
}

///|
async fn main {
  alloc_begin()
  let control = FixedArray::make(8, seed())
  keep_alive(control)
  let observed = alloc_end()
  assert_true(observed > 0UL)
  run_fixture()
  if @env.args()[1:] is ["serve-planner-fixture", _, _] {
    assert_eq(live_resources(), 0)
    assert_eq(creates(), closes())
    println(
      "MiniCPM planner fixture server: completed and all native resources closed; scripted tokens only",
    )
    return
  }
  run_vision_fixture()
  assert_eq(live_resources(), 0)
  assert_eq(creates(), closes())
  println(
    "MiniCPM-V 4.7 text ABI: queue/request/runtime startup, allocation-free steps, producing-only readback, cancellation/reuse, partial startup cleanup and active-session close passed; no GPU computation",
  )
}
