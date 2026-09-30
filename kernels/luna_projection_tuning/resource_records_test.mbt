///|
test "resource observation parser enforces scope bounds identity and valid measurements" {
  let scope = @resource_policy.Scope::new(
    device="gpu-a",
    toolchain="tool-v1",
    shape="bf16:rows32:head128",
    program="ingress-v1",
  ).unwrap()
  let prefix = "luna-projection-resources-v1\nscope\tgpu-a\ttool-v1\tbf16:rows32:head128\tingress-v1\nbudget\t128\t49152\t128\t1\n"
  let digest = "a".repeat(64)
  let record = "record\tcompact\t\{digest}\t100\t5\t64\t2\n"
  let records = parse_projection_resource_records(
    @utf8.encode(prefix + record),
    scope~,
  )
  assert_eq(records.length(), 1)
  for
    malformed in [
      record + record,
      "record\tcompact\t\{digest}\t0\t5\t64\t2\n",
      "record\tcompact\t\{digest}\t100\t2\t64\t2\n",
      "record\tcompact\t\{digest}\t100\t5\t0\t2\n",
      "record\tcompact\t\{digest}\t100\t5\t64\t0\n",
      "record\tcompact\tbad\t100\t5\t64\t2\n",
    ] {
    try
      parse_projection_resource_records(
        @utf8.encode(prefix + malformed),
        scope~,
      )
    catch {
      InvalidSnapshot => ()
      _ => fail("unexpected resource parser error")
    } noraise {
      _ => fail("invalid resource observation accepted")
    }
  }
  let wrong = @resource_policy.Scope::new(
    device="gpu-b",
    toolchain="tool-v1",
    shape="bf16:rows32:head128",
    program="ingress-v1",
  ).unwrap()
  try
    parse_projection_resource_records(
      @utf8.encode(prefix + record),
      scope=wrong,
    )
  catch {
    ScopeMismatch => ()
    _ => fail("wrong error for resource scope")
  } noraise {
    _ => fail("wrong device observation accepted")
  }
}

///|
test "machine feedback v3 is consumed by resource selection not just retained" {
  let scope = @resource_policy.Scope::new(
    device="class-a",
    toolchain="tool",
    shape="bf16",
    program="program",
  ).unwrap()
  let prefix = "luna-projection-resources-v3\nscope\tclass-a\ttool\tbf16\tprogram\nbudget\t128\t98304\t255\t1\nworkload\tprefill\t128\t1\t0\t128\t32\nprovenance\tGPU-a\nmachine-budget\t1000\t0\t100\n"
  let a = "a".repeat(64)
  let b = "b".repeat(64)
  let bytes = @utf8.encode(
    prefix +
    "record\tfast-spilling\t\{a}\t80\t5\t64\t2\t900\t16\t20\nrecord\tclean\t\{b}\t100\t5\t64\t2\t1000\t0\t100\n",
  )
  let observations = parse_projection_resource_records(bytes, scope~)
  let budget = parse_projection_resource_budget(bytes)
  let candidates = [
    @resource_policy.Candidate::new(
      @resource_policy.Policy::new(
        id="fast-spilling",
        consumer_groups=4,
        live_accumulators=2,
        operand_stages=2,
        transfer_tiles=2,
        fragment_stages=1,
      ).unwrap(),
      artifact=a,
      threads=128,
      shared_bytes=65536L,
      registers_per_thread=None,
      resident_groups=None,
    ).unwrap(),
    @resource_policy.Candidate::new(
      @resource_policy.Policy::new(
        id="clean",
        consumer_groups=4,
        live_accumulators=2,
        operand_stages=2,
        transfer_tiles=2,
        fragment_stages=1,
      ).unwrap(),
      artifact=b,
      threads=128,
      shared_bytes=65536L,
      registers_per_thread=None,
      resident_groups=None,
    ).unwrap(),
  ]
  let chosen = @resource_policy.select(
    projection_resource_selection_scope(bytes, scope~),
    budget,
    candidates,
    observations[:],
    preferred_id="fast-spilling",
  ).unwrap()
  assert_eq(chosen.policy().id(), "clean")
  // Removing feedback must not permit an optimistic, unmeasured bypass.
  assert_eq(
    @resource_policy.select(
      projection_resource_selection_scope(bytes, scope~),
      budget,
      candidates,
      [],
      preferred_id="clean",
    ),
    None,
  )
}

///|
test "workload v2 separates provenance from class and rejects mixed objectives" {
  let scope = @resource_policy.Scope::new(
    device="perf-v1:class-a",
    toolchain="tool",
    shape="bf16",
    program="program",
  ).unwrap()
  let prefix = "luna-projection-resources-v2\nscope\tperf-v1:class-a\ttool\tbf16\tprogram\nbudget\t128\t49152\t128\t1\n"
  let workload = "workload\tprefill\t128\t1\t0\t128\t32\n"
  let record = "record\tp\t" + "a".repeat(64) + "\t100\t5\t64\t2\n"
  let a = @utf8.encode(prefix + workload + "provenance\tGPU-a\n" + record)
  let b = @utf8.encode(prefix + workload + "provenance\tGPU-b\n" + record)
  assert_eq(
    parse_projection_resource_records(a, scope~),
    parse_projection_resource_records(b, scope~),
  )
  let selected = projection_resource_selection_scope(a, scope~)
  assert_true(selected != scope)
  assert_eq(selected, projection_resource_selection_scope(b, scope~))
  let other = @utf8.encode(
    prefix +
    "workload\tprefill\t128\t1\t0\t2048\t32\nprovenance\tGPU-a\n" +
    record,
  )
  assert_true(selected != projection_resource_selection_scope(other, scope~))
  for
    invalid in [
      prefix + workload + "provenance\tGPU-a\n" + workload + record,
      prefix +
      "workload\tdecode\t128\t1\t0\t128\t32\nprovenance\tGPU-a\n" +
      record,
    ] {
    try parse_projection_resource_records(@utf8.encode(invalid), scope~) catch {
      InvalidSnapshot => ()
      _ => fail("wrong error")
    } noraise {
      _ => fail("mixed or invalid objective accepted")
    }
  }
}
