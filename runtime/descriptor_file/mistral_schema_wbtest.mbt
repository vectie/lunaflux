///|
fn mistral_descriptor_fixture() -> String {
  (
    #|{"schema_version":"lunaflux.runtime.mistral_bf16.v1","model":{"family":"mistral","config_locator":"config.json","config_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","content_sha256":"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","numeric_weights_locator":"model.numeric.safetensors","numeric_weight_artifact_sha256":"2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","weight_manifest_sha256":"3123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","sliding_window_tokens":4096,"max_batch_rows":2},"kernels":{"manifest_locator":"execution.json","manifest_sha256":"4123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","policy":"deployment_approved_aot_only","admitted_bootstrap_sha256":"5123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"},"execution":{"device_ordinal":0,"compute_major":8,"compute_minor":9,"supports_bf16":true,"supports_cublas_lt":true,"tokens_per_page":16,"total_page_count":4096,"model_generation":1},"worker_limits":{"max_prefill_rows":2,"max_decode_rows":2,"max_plan_rows":2,"max_plan_tokens":16,"max_plan_pages":4,"max_capabilities":16,"max_completion_slots":2,"max_sequence_tokens":16,"max_token_id":31},"inference_limits":{"max_text_bytes":1024,"max_input_tokens":16,"max_new_tokens":16,"max_context_tokens":16,"max_token_id":31,"max_stop_token_ids":4,"max_stop_strings":4,"max_stop_string_bytes":32,"max_trace_bytes":32,"max_cache_scope_bytes":32,"max_decoded_delta_bytes":64,"max_deadline_millis":60000,"max_top_k":8,"max_temperature":2.0},"ceilings":{"max_model_config_bytes":4096,"max_weight_file_bytes":4096,"max_weight_arena_bytes":4096,"max_activation_arena_bytes":8192,"max_kv_arena_bytes":8192,"max_execution_manifest_bytes":4096,"max_module_bytes":4096,"max_total_module_bytes":4096,"max_graph_capture_bytes":16384}}
  )
}

///|
fn parse_mistral_fixture(
  text : String,
) -> MistralRuntimeDescriptorClaims raise RuntimeDescriptorFileError {
  parse_mistral_descriptor(
    @utf8.encode(text),
    RuntimeDescriptorFileLimits::new(
      max_descriptor_bytes=16384L,
      max_json_depth=16,
    ),
  )
}

///|
test "Mistral descriptor retains exact family artifact manifest and window" {
  let claims = parse_mistral_fixture(mistral_descriptor_fixture())
  assert_eq(claims.model.family, "mistral")
  assert_eq(claims.model.numeric_weights_locator, "model.numeric.safetensors")
  assert_eq(
    claims.model.numeric_weight_artifact_digest,
    "2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  assert_eq(
    claims.model.weight_manifest_digest,
    "3123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  assert_eq(claims.model.sliding_window_tokens, 4096)
  assert_eq(claims.model.max_batch_rows, 2)
}

///|
test "Mistral descriptor rejects family schema and field substitution" {
  let canonical = mistral_descriptor_fixture()
  for
    text in [
      canonical.replace(
        old="\"family\":\"mistral\"",
        new="\"family\":\"llama\"",
      ),
      canonical.replace(
        old="lunaflux.runtime.mistral_bf16.v1",
        new="lunaflux.runtime.v1",
      ),
      canonical.replace(
        old="\"weight_manifest_sha256\":",
        new="\"ambient_manifest\":0,\"weight_manifest_sha256\":",
      ),
    ] {
    try {
      let claims = parse_mistral_fixture(text)
      if claims.model.family != "mistral" {
        raise AdmissionRejected(ModelConfig)
      }
    } catch {
      InvalidField(_) | UnknownField(_) | AdmissionRejected(ModelConfig) => ()
      _ => fail("Mistral schema substitution reported an unrelated failure")
    } noraise {
      _ => fail("Mistral schema substitution was admitted")
    }
  }
}

///|
test "Mistral descriptor rejects incomplete and duplicate identity fields" {
  let canonical = mistral_descriptor_fixture()
  for
    text in [
      canonical.replace(old="\"sliding_window_tokens\":4096,", new=""),
      canonical.replace(
        old="\"weight_manifest_sha256\":\"3123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"",
        new="\"weight_manifest_sha256\":\"3123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\",\"weight_manifest_sha256\":\"3123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"",
      ),
    ] {
    try parse_mistral_fixture(text) catch {
      MissingField(_) | DuplicateField(..) => ()
      _ => fail("incomplete Mistral identity reported an unrelated failure")
    } noraise {
      _ => fail("incomplete or duplicate Mistral identity was admitted")
    }
  }
}

///|
test "Mistral v1 remains standalone and v2 owns optional startup extensions" {
  let v1 = parse_mistral_fixture(mistral_descriptor_fixture())
  if v1.kernels.fused_runtime is Some(_) {
    fail("Mistral v1 gained a fused runtime source")
  }
  assert_eq(v1.kernels.sampling_runtime, HostSamplingRuntime)
  let v2 = mistral_descriptor_fixture()
    .replace(
      old="lunaflux.runtime.mistral_bf16.v1",
      new="lunaflux.runtime.mistral_bf16.v2",
    )
    .replace(
      old="\"admitted_bootstrap_sha256\":\"5123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"",
      new="\"admitted_bootstrap_sha256\":\"5123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\",\"fused_runtime\":{\"locator\":\"fused.runtime.v1\",\"sha256\":\"6123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"}",
    )
  guard parse_mistral_fixture(v2).kernels.fused_runtime is Some(fused) else {
    fail("Mistral v2 lost its fused runtime source")
  }
  assert_eq(fused.locator, "fused.runtime.v1")
  try
    parse_mistral_fixture(
      v2.replace(
        old="lunaflux.runtime.mistral_bf16.v2",
        new="lunaflux.runtime.mistral_bf16.v1",
      ),
    )
  catch {
    InvalidField(area=Kernels, issue=UnsupportedValue, index=_) => ()
    _ => fail("unexpected Mistral v1 extension failure")
  } noraise {
    _ => fail("Mistral v1 silently accepted v2 startup extensions")
  }
}
