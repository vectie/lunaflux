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
