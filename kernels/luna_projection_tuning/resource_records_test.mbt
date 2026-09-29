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
