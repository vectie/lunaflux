///|
test "exact workload and bucket keep independent objectives and lawful fallback" {
  let base = @resource_policy.Scope::new(
    device="sm121-sm40-cache24-memory128-powerA",
    toolchain="compiler",
    shape="bf16-layout-v1",
    program="source",
  ).unwrap()
  let small = @resource_policy.Workload::new(
    phase="prefill",
    query_tokens=128,
    batch_rows=1,
    prior_history=0,
    bucket_tokens=128,
    bucket_rows=1,
  ).unwrap()
  let large = @resource_policy.Workload::new(
    phase="prefill",
    query_tokens=2048,
    batch_rows=8,
    prior_history=2048,
    bucket_tokens=2048,
    bucket_rows=8,
  ).unwrap()
  let envelope = @resource_policy.Workload::new(
    phase="prefill",
    query_tokens=128,
    batch_rows=1,
    prior_history=0,
    bucket_tokens=2048,
    bucket_rows=32,
  ).unwrap()
  assert_true(small.scope(base) != large.scope(base))
  assert_true(small.scope(base) != envelope.scope(base))
  let policy = @resource_policy.Policy::new(
    id="default",
    consumer_groups=4,
    live_accumulators=2,
    operand_stages=2,
    transfer_tiles=1,
    fragment_stages=1,
  ).unwrap()
  let candidate = @resource_policy.Candidate::new(
    policy,
    artifact="artifact",
    threads=128,
    shared_bytes=0L,
    registers_per_thread=None,
    resident_groups=None,
  ).unwrap()
  let budget = @resource_policy.ResourceBudget::new(
    max_threads=128,
    max_shared_bytes=49152L,
    max_registers_per_thread=None,
    min_resident_groups=None,
  ).unwrap()
  let observation = @resource_policy.Observation::new(
    large.scope(base).unwrap(),
    policy_id="default",
    artifact="artifact",
    latency_ns=100L,
    samples=5,
    registers_per_thread=32,
    resident_groups=2,
  ).unwrap()
  assert_eq(
    @resource_policy.select(
      small.scope(base).unwrap(),
      budget,
      [candidate],
      [observation],
      preferred_id="default",
    )
    .unwrap()
    .origin(),
    UnmeasuredPreferred,
  )
  assert_eq(
    @resource_policy.select(
      large.scope(base).unwrap(),
      budget,
      [candidate],
      [observation],
      preferred_id="default",
    )
    .unwrap()
    .origin(),
    Measured(latency_ns=100L, samples=5),
  )
  assert_eq(
    @resource_policy.Workload::new(
      phase="decode",
      query_tokens=8,
      batch_rows=1,
      prior_history=0,
      bucket_tokens=8,
      bucket_rows=1,
    ),
    None,
  )
  let alternate = @resource_policy.Candidate::new(
    @resource_policy.Policy::new(
      id="alternate",
      consumer_groups=4,
      live_accumulators=1,
      operand_stages=2,
      transfer_tiles=2,
      fragment_stages=1,
    ).unwrap(),
    artifact="alternate-artifact",
    threads=128,
    shared_bytes=0L,
    registers_per_thread=None,
    resident_groups=None,
  ).unwrap()
  let measurements : Array[@resource_policy.Observation] = [observation]
  for
    (workload, id, artifact, ns) in [
      (small, "default", "artifact", 90L),
      (small, "alternate", "alternate-artifact", 100L),
      (large, "alternate", "alternate-artifact", 80L),
    ] {
    measurements.push(
      @resource_policy.Observation::new(
        workload.scope(base).unwrap(),
        policy_id=id,
        artifact~,
        latency_ns=ns,
        samples=5,
        registers_per_thread=32,
        resident_groups=2,
      ).unwrap(),
    )
  }
  for (workload, expected) in [(small, "default"), (large, "alternate")] {
    for candidates in [[candidate, alternate], [alternate, candidate]] {
      assert_eq(
        @resource_policy.select(
          workload.scope(base).unwrap(),
          budget,
          candidates,
          measurements,
          preferred_id="default",
        )
        .unwrap()
        .policy()
        .id(),
        expected,
      )
    }
  }
}
