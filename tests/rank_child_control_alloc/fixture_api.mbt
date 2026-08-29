///|
extern "C" fn alloc_probe_begin() = "lunaflux_alloc_probe_begin"

///|
extern "C" fn alloc_probe_end() -> UInt64 = "lunaflux_alloc_probe_end"

///|
extern "C" fn alloc_probe_check(observed : UInt64, expect_zero : Int) -> Int = "lunaflux_alloc_probe_check"

///|
extern "C" fn alloc_probe_abort() = "lunaflux_alloc_probe_abort"

///|
extern "C" fn alloc_probe_seed() -> Int = "lunaflux_alloc_probe_seed"

///|
#borrow(value)
extern "C" fn keep_array_alive(value : FixedArray[Int]) = "lunaflux_alloc_probe_keep_alive"

///|
extern "C" fn fixture_begin() -> Int = "lunaflux_rank_child_fixture_begin"

///|
#borrow(source)
extern "C" fn fixture_send(source : FixedArray[Byte], length : Int) -> Int = "lunaflux_rank_child_fixture_send"

///|
#borrow(destination)
extern "C" fn fixture_receive(destination : FixedArray[Byte]) -> Int = "lunaflux_rank_child_fixture_receive"

///|
extern "C" fn fixture_shutdown_write() -> Int = "lunaflux_rank_child_fixture_shutdown_write"

///|
extern "C" fn fixture_finish() -> Int = "lunaflux_rank_child_fixture_finish"

///|
extern "C" fn fixture_reset_evidence() = "lunaflux_rank_child_fixture_reset_evidence"

///|
extern "C" fn fixture_check_evidence(
  expected_mask : UInt,
  minimum_count : UInt64,
) -> Int = "lunaflux_rank_child_fixture_check_evidence"

///|
fn fixture_identity() -> @spec.ModelIdentity {
  let content = try! @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let plan = try! @spec.PlanDigest::from_sha256(
    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
  )
  try! @spec.ModelIdentity::new(content, plan)
}

///|
fn fixture_binding() -> @rank_group_wire.RankGroupWireBinding {
  try! @rank_group_wire.RankGroupWireBinding::new(
    identity=fixture_identity(),
    model_generation_value=1UL,
    group_generation_value=1UL,
    rank=0,
    world_size=2,
    group_digest="1111111111111111111111111111111111111111111111111111111111111111",
    topology_digest="2222222222222222222222222222222222222222222222222222222222222222",
    worker_digest="3333333333333333333333333333333333333333333333333333333333333333",
    device_plan_digest="4444444444444444444444444444444444444444444444444444444444444444",
    execution_plan_digest="9999999999999999999999999999999999999999999999999999999999999999",
    collective_digest="5555555555555555555555555555555555555555555555555555555555555555",
    kv_digest="6666666666666666666666666666666666666666666666666666666666666666",
  )
}

///|
fn verify_positive_control() -> Unit {
  alloc_probe_begin()
  let allocated = FixedArray::make(8, alloc_probe_seed())
  keep_array_alive(allocated)
  if alloc_probe_check(alloc_probe_end(), 0) != 1 {
    alloc_probe_abort()
  }
}
