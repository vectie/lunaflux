///|
fn i8_descriptor_fixture() -> String {
  (
    #|{"schema_version":"lunaflux.runtime.i8.v2","model":{"config_locator":"config.json","config_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","content_sha256":"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","numeric_weights_locator":"model.numeric.safetensors","numeric_weight_artifact_sha256":"2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","max_batch_rows":1},"kernels":{"manifest_locator":"execution-i8-v4.json","manifest_sha256":"3123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","policy":"production_symmetric_i8_weight_only_per_output_channel_f32_scale_f32_accumulate_bf16_output_v1","admitted_bootstrap_v3_sha256":"4123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"},"execution":{"device_ordinal":0,"compute_major":8,"compute_minor":9,"total_memory_bytes":25769803776,"supports_bf16":true,"supports_cublas_lt":true,"tokens_per_page":16,"total_page_count":4096,"model_generation":1},"worker_limits":{"max_prefill_rows":1,"max_decode_rows":1,"max_plan_rows":1,"max_plan_tokens":8,"max_plan_pages":2,"max_capabilities":16,"max_completion_slots":1,"max_sequence_tokens":8,"max_token_id":7},"inference_limits":{"max_text_bytes":1024,"max_input_tokens":8,"max_new_tokens":8,"max_context_tokens":8,"max_token_id":7,"max_stop_token_ids":4,"max_stop_strings":4,"max_stop_string_bytes":32,"max_trace_bytes":32,"max_cache_scope_bytes":32,"max_decoded_delta_bytes":64,"max_deadline_millis":60000,"max_top_k":8,"max_temperature":2.0},"ceilings":{"max_model_config_bytes":4096,"max_weight_file_bytes":4096,"max_weight_arena_bytes":4096,"max_activation_arena_bytes":8192,"max_kv_arena_bytes":8192,"max_execution_manifest_bytes":4096,"max_module_bytes":4096,"max_total_module_bytes":4096}}
  )
}

///|
fn parse_i8_fixture(text : String) -> I8RuntimeDescriptorClaims raise {
  parse_i8_descriptor(
    @utf8.encode(text),
    RuntimeDescriptorFileLimits::new(
      max_descriptor_bytes=16384L,
      max_json_depth=16,
    ),
  )
}

///|
test "I8 runtime v2 schema retains exact numeric, bootstrap, and device facts" {
  let claims = parse_i8_fixture(i8_descriptor_fixture())
  assert_eq(claims.model.numeric_weights_locator, "model.numeric.safetensors")
  assert_eq(
    claims.model.numeric_weight_artifact_digest,
    "2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  assert_eq(claims.execution.compute_major, 8)
  assert_eq(claims.execution.compute_minor, 9)
  assert_eq(claims.execution.total_memory_bytes, 25769803776L)
  assert_true(claims.execution.supports_bf16)
  assert_true(claims.execution.supports_cublas_lt)
}

///|
test "I8 runtime v2 rejects legacy schema, policy, and unknown fields" {
  let canonical = i8_descriptor_fixture()
  for
    text in [
      canonical.replace(old="lunaflux.runtime.i8.v2", new="lunaflux.runtime.v1"),
      canonical.replace(
        old=I8_RUNTIME_POLICY,
        new="deployment_approved_aot_only",
      ),
      canonical.replace(
        old="\"max_batch_rows\":1",
        new="\"max_batch_rows\":1,\"unknown\":0",
      ),
    ] {
    try parse_i8_fixture(text) catch {
      InvalidField(_) | UnknownField(_) => ()
      _ => fail("unexpected strict I8 runtime schema failure")
    } noraise {
      _ => fail("legacy or unknown I8 descriptor input was admitted")
    }
  }
}

///|
test "I8 runtime v2 device facts admit exact supported I8 targets" {
  let claims = parse_i8_fixture(i8_descriptor_fixture())
  let (_, numeric, target) = admit_i8_device_capability(claims.execution)
  assert_eq(target.compute_major(), 8)
  assert_eq(target.compute_minor(), 9)
  assert_eq(
    numeric.feature(),
    SymmetricI8WeightOnlyPerOutputChannelF32ScaleF32AccumulateBf16OutputV1,
  )
  let blackwell : I8RuntimeDescriptorExecutionClaims = {
    ..claims.execution,
    compute_major: 12,
    compute_minor: 0,
  }
  let (_, blackwell_numeric, blackwell_target) = admit_i8_device_capability(
    blackwell,
  )
  assert_eq(blackwell_target.compute_major(), 12)
  assert_eq(blackwell_target.compute_minor(), 0)
  assert_eq(
    blackwell_numeric.feature(),
    SymmetricI8WeightOnlyPerOutputChannelF32ScaleF32AccumulateBf16OutputV1,
  )
  let unsupported : I8RuntimeDescriptorExecutionClaims = {
    ..claims.execution,
    compute_minor: 0,
  }
  try admit_i8_device_capability(unsupported) catch {
    AdmissionRejected(DeviceCapability) => ()
    _ => fail("unexpected unsupported I8 device failure")
  } noraise {
    _ => fail("unsupported compute 8.0 entered I8 admission")
  }
}
