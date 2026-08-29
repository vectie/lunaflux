///|
fn tensor_parallel_descriptor_fixture() -> String {
  let base = graph_descriptor_fixture()
    .replace(
      old="lunaflux.runtime.v1",
      new="lunaflux.runtime.tensor-parallel.v1",
    )
    .replace(
      old=",\"admitted_bootstrap_sha256\":\"3123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"",
      new="",
    )
  base[:base.length() - 1].to_owned() +
  ",\"tensor_parallel\":{\"ranks\":[{\"rank\":0,\"device_ordinal\":0,\"memory_ceiling_bytes\":8192,\"weight_reserved_bytes\":2048,\"activation_workspace_reserved_bytes\":1024,\"kv_reserved_bytes\":4096,\"collective_reserved_bytes\":1024},{\"rank\":1,\"device_ordinal\":1,\"memory_ceiling_bytes\":8192,\"weight_reserved_bytes\":2048,\"activation_workspace_reserved_bytes\":1024,\"kv_reserved_bytes\":4096,\"collective_reserved_bytes\":1024}],\"minimum_collective_runtime_version_code\":21403,\"maximum_collective_runtime_version_code\":21999,\"collective_exchange_timeout_millis\":600000}}"
}

///|
fn parse_tensor_parallel_fixture(
  text : String,
) -> TensorParallelRuntimeDescriptorClaims raise RuntimeDescriptorFileError {
  parse_tensor_parallel_descriptor(
    @utf8.encode(text),
    RuntimeDescriptorFileLimits::new(
      max_descriptor_bytes=65536L,
      max_json_depth=16,
    ),
  )
}

///|
test "tensor parallel schema retains exact ordered rank and collective claims" {
  let claims = try! parse_tensor_parallel_fixture(
    tensor_parallel_descriptor_fixture(),
  )
  assert_eq(claims.tensor_parallel.ranks.length(), 2)
  assert_eq(claims.tensor_parallel.ranks[0].rank, 0)
  assert_eq(claims.tensor_parallel.ranks[1].device_ordinal, 1)
  assert_eq(
    claims.tensor_parallel.minimum_collective_runtime_version_code,
    21403,
  )
  assert_eq(claims.tensor_parallel.collective_exchange_timeout_millis, 600000)
}

///|
test "tensor parallel schema rejects reordered ranks and legacy bootstrap digest" {
  let canonical = tensor_parallel_descriptor_fixture()
  let hostile = [
    canonical.replace(
      old="\"rank\":1,\"device_ordinal\":1",
      new="\"rank\":0,\"device_ordinal\":1",
    ),
    canonical.replace(
      old="\"policy\":\"deployment_approved_aot_only\"",
      new="\"policy\":\"deployment_approved_aot_only\",\"admitted_bootstrap_sha256\":\"3123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"",
    ),
  ]
  for text in hostile {
    try parse_tensor_parallel_fixture(text) catch {
      InvalidField(..) | UnknownField(..) => ()
      _ => fail("unexpected tensor-parallel schema rejection")
    } noraise {
      _ => fail("noncanonical tensor-parallel schema was admitted")
    }
  }
}
