///|
fn graph_descriptor_fixture() -> String {
  let text =
    #|{"schema_version":"lunaflux.runtime.v1","model":{"config_locator":"config.json","config_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","weights_locator":"model.safetensors","content_sha256":"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","max_batch_rows":1},"kernels":{"manifest_locator":"execution.json","manifest_sha256":"2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","policy":"deployment_approved_aot_only","admitted_bootstrap_sha256":"3123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"},"execution":{"device_ordinal":0,"compute_major":8,"compute_minor":0,"supports_bf16":true,"supports_cublas_lt":false,"tokens_per_page":4,"total_page_count":8,"model_generation":1},"worker_limits":{"max_prefill_rows":1,"max_decode_rows":1,"max_plan_rows":1,"max_plan_tokens":8,"max_plan_pages":2,"max_capabilities":16,"max_completion_slots":1,"max_sequence_tokens":8,"max_token_id":7},"inference_limits":{"max_text_bytes":1024,"max_input_tokens":8,"max_new_tokens":8,"max_context_tokens":8,"max_token_id":7,"max_stop_token_ids":4,"max_stop_strings":4,"max_stop_string_bytes":32,"max_trace_bytes":32,"max_cache_scope_bytes":32,"max_decoded_delta_bytes":64,"max_deadline_millis":60000,"max_top_k":8,"max_temperature":2.0},"ceilings":{"max_model_config_bytes":4096,"max_weight_file_bytes":4096,"max_weight_arena_bytes":4096,"max_activation_arena_bytes":8192,"max_kv_arena_bytes":8192,"max_execution_manifest_bytes":4096,"max_module_bytes":4096,"max_total_module_bytes":4096}}
  text
}

///|
fn parse_graph_descriptor_fixture(
  text : String,
) -> RuntimeDescriptorClaims raise {
  parse_descriptor(
    @utf8.encode(text),
    RuntimeDescriptorFileLimits::new(
      max_descriptor_bytes=16384L,
      max_json_depth=16,
    ),
  )
}

///|
test "descriptor v1 preserves explicit absent graph ceiling" {
  let claims = parse_graph_descriptor_fixture(graph_descriptor_fixture())
  assert_eq(claims.ceilings.max_graph_capture_bytes, None)
  assert_eq(admit_graph_capture_ceiling(claims.ceilings), None)
}

///|
test "descriptor v2 requires and parses one positive graph ceiling" {
  let canonical = graph_descriptor_fixture()
    .replace(old="lunaflux.runtime.v1", new="lunaflux.runtime.v2")
    .replace(
      old="\"max_total_module_bytes\":4096",
      new="\"max_total_module_bytes\":4096,\"max_graph_capture_bytes\":8192",
    )
  let claims = parse_graph_descriptor_fixture(canonical)
  assert_eq(claims.ceilings.max_graph_capture_bytes, Some(8192L))
  assert_eq(admit_graph_capture_ceiling(claims.ceilings), Some(8192L))

  let missing = canonical.replace(
    old=",\"max_graph_capture_bytes\":8192",
    new="",
  )
  try parse_graph_descriptor_fixture(missing) catch {
    MissingField(area=Ceilings, index=8) => ()
    _ => fail("unexpected missing graph ceiling failure")
  } noraise {
    _ => fail("descriptor v2 without a graph ceiling was admitted")
  }

  let zero = canonical.replace(
    old="\"max_graph_capture_bytes\":8192",
    new="\"max_graph_capture_bytes\":0",
  )
  let zero_claims = parse_graph_descriptor_fixture(zero)
  try admit_graph_capture_ceiling(zero_claims.ceilings) catch {
    AdmissionRejected(GraphMemory) => ()
    _ => fail("unexpected zero graph ceiling failure")
  } noraise {
    _ => fail("zero graph ceiling was admitted")
  }
}

///|
test "descriptor v1 rejects an unversioned graph ceiling" {
  let hostile = graph_descriptor_fixture().replace(
    old="\"max_total_module_bytes\":4096",
    new="\"max_total_module_bytes\":4096,\"max_graph_capture_bytes\":8192",
  )
  try parse_graph_descriptor_fixture(hostile) catch {
    UnknownField(area=Ceilings, index=8) => ()
    _ => fail("unexpected v1 graph ceiling failure")
  } noraise {
    _ => fail("v1 silently accepted the v2 graph ceiling")
  }
}

///|
test "descriptor v3 remains standalone and v4 owns optional startup extensions" {
  let v3 = graph_descriptor_fixture()
    .replace(old="lunaflux.runtime.v1", new="lunaflux.runtime.v3")
    .replace(
      old="\"max_total_module_bytes\":4096",
      new="\"max_total_module_bytes\":4096,\"max_graph_capture_bytes\":8192",
    )
  let legacy = parse_graph_descriptor_fixture(v3)
  if legacy.kernels.fused_runtime is Some(_) {
    fail("descriptor v3 gained a fused runtime source")
  }
  assert_eq(legacy.kernels.sampling_runtime, HostSamplingRuntime)
  let v4 = v3
    .replace(old="lunaflux.runtime.v3", new="lunaflux.runtime.v4")
    .replace(
      old="\"admitted_bootstrap_sha256\":\"3123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"",
      new="\"admitted_bootstrap_sha256\":\"3123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\",\"fused_runtime\":{\"locator\":\"fused.runtime.v1\",\"sha256\":\"4123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"},\"sampling_runtime\":\"embedded_cuda_greedy_v1\"",
    )
  let extended = parse_graph_descriptor_fixture(v4)
  guard extended.kernels.fused_runtime is Some(fused) else {
    fail("descriptor v4 lost its fused runtime source")
  }
  assert_eq(fused.locator, "fused.runtime.v1")
  assert_eq(extended.kernels.sampling_runtime, EmbeddedCudaGreedyV1)
  try
    parse_graph_descriptor_fixture(
      v4.replace(old="lunaflux.runtime.v4", new="lunaflux.runtime.v3"),
    )
  catch {
    InvalidField(area=Kernels, issue=UnsupportedValue, index=_) => ()
    _ => fail("unexpected descriptor v3 extension failure")
  } noraise {
    _ => fail("descriptor v3 silently accepted v4 startup extensions")
  }
}
