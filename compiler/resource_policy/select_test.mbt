///|
fn scope(
  device? : String = "device-a",
  shape? : String = "bf16:32x64x128",
  program? : String = "source-v1",
  toolchain? : String = "tool-v1",
) -> @resource_policy.Scope {
  @resource_policy.Scope::new(device~, toolchain~, shape~, program~).unwrap()
}

///|
fn candidate(
  id : String,
  shared? : Int64 = 1024L,
  registers? : Int? = Some(32),
  residency? : Int? = Some(2),
) -> @resource_policy.Candidate {
  let policy = @resource_policy.Policy::new(
    id~,
    consumer_groups=4,
    live_accumulators=2,
    operand_stages=2,
    transfer_tiles=4,
    fragment_stages=1,
  ).unwrap()
  @resource_policy.Candidate::new(
    policy,
    artifact=id + "-code-v1",
    threads=128,
    shared_bytes=shared,
    registers_per_thread=registers,
    resident_groups=residency,
  ).unwrap()
}

///|
fn observation(
  scope : @resource_policy.Scope,
  id : String,
  ns : Int64,
  artifact? : String = id + "-code-v1",
  registers? : Int = 32,
  residency? : Int = 2,
) -> @resource_policy.Observation {
  @resource_policy.Observation::new(
    scope,
    policy_id=id,
    artifact~,
    latency_ns=ns,
    samples=5,
    registers_per_thread=registers,
    resident_groups=residency,
  ).unwrap()
}

///|
fn budget() -> @resource_policy.ResourceBudget {
  @resource_policy.ResourceBudget::new(
    max_threads=256,
    max_shared_bytes=4096L,
    max_registers_per_thread=Some(64),
    min_resident_groups=Some(2),
  ).unwrap()
}

///|
test "measurement scope includes device toolchain exact shape source and artifact" {
  let good = scope()
  let a = candidate("a")
  let b = candidate("b")
  for
    wrong in [
      scope(device="other"),
      scope(toolchain="other"),
      scope(shape="bf16:64x64x128"),
      scope(program="source-v2"),
    ] {
    let result = @resource_policy.select(
      good,
      budget(),
      [a, b],
      [observation(wrong, "b", 1L)],
      preferred_id="a",
    ).unwrap()
    assert_eq(result.policy().id(), "a")
    assert_eq(result.origin(), @resource_policy.UnmeasuredPreferred)
  }
  let result = @resource_policy.select(
    good,
    budget(),
    [a, b],
    [observation(good, "b", 1L, artifact="b-code-v0")],
    preferred_id="a",
  ).unwrap()
  assert_eq(result.policy().id(), "a")
  assert_eq(result.origin(), @resource_policy.UnmeasuredPreferred)
}

///|
test "measured winner and ties are stable under input permutation" {
  let s = scope()
  let records = [observation(s, "b", 20L), observation(s, "a", 20L)]
  for
    candidates in [
      [candidate("a"), candidate("b")],
      [candidate("b"), candidate("a")],
    ] {
    let result = @resource_policy.select(
      s,
      budget(),
      candidates,
      records,
      preferred_id="b",
    ).unwrap()
    assert_eq(result.policy().id(), "a")
    assert_eq(
      result.origin(),
      @resource_policy.Measured(latency_ns=20L, samples=5),
    )
  }
  let result = @resource_policy.select(
    s,
    budget(),
    [candidate("a"), candidate("b")],
    [observation(s, "a", 30L), observation(s, "b", 10L)],
    preferred_id="a",
  ).unwrap()
  assert_eq(result.policy().id(), "b")
}

///|
test "resource limits reject fast but infeasible candidates without optimistic fallback" {
  let s = scope()
  let candidates = [
    candidate("a"),
    candidate("b"),
    candidate("c", shared=4097L),
  ]
  for
    records in [
      [observation(s, "b", 1L, registers=65)],
      [observation(s, "b", 1L, residency=1)],
    ] {
    let result = @resource_policy.select(
      s,
      budget(),
      candidates,
      records,
      preferred_id="b",
    ).unwrap()
    assert_eq(result.policy().id(), "a")
    assert_eq(result.origin(), @resource_policy.UnmeasuredEligible)
  }
  assert_true(
    @resource_policy.select(
      s,
      budget(),
      [candidate("c", shared=4097L)],
      [observation(s, "c", 1L)],
      preferred_id="c",
    )
    is None,
  )
}

///|
test "unknown constrained resources need observations and missing data is labelled" {
  let s = scope()
  let unknown = candidate("a", registers=None, residency=None)
  assert_true(
    @resource_policy.select(s, budget(), [unknown], [], preferred_id="a")
    is None,
  )
  let result = @resource_policy.select(
    s,
    budget(),
    [unknown],
    [observation(s, "a", 100L)],
    preferred_id="a",
  ).unwrap()
  assert_true(result.origin() is Measured(..))
  let unconstrained = @resource_policy.ResourceBudget::new(
    max_threads=128,
    max_shared_bytes=1024L,
    max_registers_per_thread=None,
    min_resident_groups=None,
  ).unwrap()
  assert_eq(
    @resource_policy.select(s, unconstrained, [unknown], [], preferred_id="a")
    .unwrap()
    .origin(),
    @resource_policy.UnmeasuredPreferred,
  )
}

///|
test "duplicate and oversized offline sets cannot produce order dependent winners" {
  let s = scope()
  let a = candidate("a")
  let r = observation(s, "a", 100L)
  assert_true(
    @resource_policy.select(s, budget(), [a, a], [], preferred_id="a") is None,
  )
  assert_true(
    @resource_policy.select(s, budget(), [a], [r, r], preferred_id="a") is None,
  )
  assert_true(
    @resource_policy.select(s, budget(), [], [], preferred_id="a") is None,
  )
  assert_true(
    @resource_policy.select(
      s,
      budget(),
      [a],
      Array::make(4097, r),
      preferred_id="a",
    )
    is None,
  )
}

///|
test "invalid resource policy geometry is not admitted" {
  assert_true(
    @resource_policy.Policy::new(
      id="bad",
      consumer_groups=0,
      live_accumulators=2,
      operand_stages=2,
      transfer_tiles=4,
      fragment_stages=1,
    )
    is None,
  )
  assert_true(
    @resource_policy.Policy::new(
      id="bad",
      consumer_groups=4,
      live_accumulators=2,
      operand_stages=2,
      transfer_tiles=1,
      fragment_stages=2,
    )
    is None,
  )
  assert_true(
    @resource_policy.Observation::new(
      scope(),
      policy_id="a",
      artifact="a-code",
      latency_ns=1L,
      samples=2,
      registers_per_thread=32,
      resident_groups=2,
    )
    is None,
  )
}

///|
test "offline budgets can narrow but never widen a backend launch envelope" {
  let backend = @resource_policy.ResourceBudget::new(
    max_threads=128,
    max_shared_bytes=1024L,
    max_registers_per_thread=Some(64),
    min_resident_groups=Some(2),
  ).unwrap()
  let permissive = @resource_policy.ResourceBudget::new(
    max_threads=1024,
    max_shared_bytes=100000L,
    max_registers_per_thread=None,
    min_resident_groups=None,
  ).unwrap()
  assert_eq(backend.intersect(permissive), backend)
  assert_eq(permissive.intersect(backend), backend)
  let restrictive = @resource_policy.ResourceBudget::new(
    max_threads=64,
    max_shared_bytes=512L,
    max_registers_per_thread=Some(32),
    min_resident_groups=Some(3),
  ).unwrap()
  assert_eq(backend.intersect(restrictive), restrictive)
  let scope = scope()
  assert_true(
    @resource_policy.select(
      scope,
      backend.intersect(restrictive),
      [candidate("a")],
      [observation(scope, "a", 1L)],
      preferred_id="a",
    )
    is None,
  )
}
