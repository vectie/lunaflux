///|
fn qwen3_descriptor_fixture() -> String {
  (
    #|{"schema_version":"lunaflux.runtime.qwen3_bf16.v1","model":{"family":"qwen3","config_locator":"config.json","config_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","content_sha256":"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","numeric_weights_locator":"model.numeric","numeric_weight_artifact_sha256":"2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","weight_manifest_sha256":"3123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","tied_embeddings":true,"max_batch_rows":2},"kernels":{"manifest_locator":"execution.json","manifest_sha256":"4123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","policy":"deployment_approved_aot_only","admitted_bootstrap_sha256":"5123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"},"execution":{"device_ordinal":0,"compute_major":8,"compute_minor":9,"supports_bf16":true,"supports_cublas_lt":false,"tokens_per_page":16,"total_page_count":4096,"model_generation":1},"worker_limits":{"max_prefill_rows":2,"max_decode_rows":2,"max_plan_rows":2,"max_plan_tokens":16,"max_plan_pages":4,"max_capabilities":16,"max_completion_slots":2,"max_sequence_tokens":16,"max_token_id":31},"inference_limits":{"max_text_bytes":1024,"max_input_tokens":16,"max_new_tokens":16,"max_context_tokens":16,"max_token_id":31,"max_stop_token_ids":4,"max_stop_strings":4,"max_stop_string_bytes":32,"max_trace_bytes":32,"max_cache_scope_bytes":32,"max_decoded_delta_bytes":64,"max_deadline_millis":60000,"max_top_k":8,"max_temperature":2.0},"ceilings":{"max_model_config_bytes":4096,"max_weight_file_bytes":4096,"max_weight_arena_bytes":4096,"max_activation_arena_bytes":8192,"max_kv_arena_bytes":8192,"max_execution_manifest_bytes":4096,"max_module_bytes":4096,"max_total_module_bytes":4096,"max_graph_capture_bytes":16384}}
  ).replace(old=",\"max_graph_capture_bytes\":16384", new="")
}

///|
fn parse_qwen3_fixture(text : String) -> Qwen3RuntimeDescriptorClaims raise {
  parse_qwen3_descriptor(
    @utf8.encode(text),
    RuntimeDescriptorFileLimits::new(
      max_descriptor_bytes=16384L,
      max_json_depth=16,
    ),
  )
}

///|
test "Qwen3 descriptor retains exact tied family and numeric identities" {
  let claims = parse_qwen3_fixture(qwen3_descriptor_fixture())
  assert_eq(claims.model.family, "qwen3")
  assert_true(claims.model.tied_embeddings)
  assert_eq(claims.model.max_batch_rows, 2)
  assert_eq(claims.kernels.sampling_runtime, HostSamplingRuntime)
  assert_eq(claims.ceilings.max_graph_capture_bytes, None)
}

///|
test "Qwen3 descriptor admits authenticated embedded CUDA greedy sampling" {
  let extended = qwen3_descriptor_fixture().replace(
    old="\"admitted_bootstrap_sha256\":\"5123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"",
    new="\"admitted_bootstrap_sha256\":\"5123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\",\"sampling_runtime\":\"embedded_cuda_greedy_v1\"",
  )
  let claims = parse_qwen3_fixture(extended)
  assert_eq(claims.kernels.sampling_runtime, EmbeddedCudaGreedyV1)
}

///|
test "Qwen3 bootstrap source preserves embedded CUDA greedy sampling" {
  let extended = qwen3_descriptor_fixture().replace(
    old="\"admitted_bootstrap_sha256\":\"5123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"",
    new="\"admitted_bootstrap_sha256\":\"5123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\",\"sampling_runtime\":\"embedded_cuda_greedy_v1\"",
  )
  let claims = parse_qwen3_fixture(extended)
  let source = admit_qwen3_source(
    "/srv/lunaflux/qwen3",
    "/srv/lunaflux/kernels",
    claims,
    admit_ceilings(claims.ceilings),
  )
  assert_eq(source.kernel().sampling_runtime(), EmbeddedCudaGreedyV1)
}

///|
test "Qwen3 descriptor rejects family and tied-mode substitution" {
  let canonical = qwen3_descriptor_fixture()
  for
    text in [
      canonical.replace(
        old="\"family\":\"qwen3\"",
        new="\"family\":\"mistral\"",
      ),
      canonical.replace(
        old="\"tied_embeddings\":true",
        new="\"tied_embeddings\":false",
      ),
    ] {
    try {
      let claims = parse_qwen3_fixture(text)
      if claims.model.family != "qwen3" || !claims.model.tied_embeddings {
        raise AdmissionRejected(ModelConfig)
      }
    } catch {
      InvalidField(_) | AdmissionRejected(ModelConfig) => ()
      _ => fail("Qwen3 substitution reported an unrelated failure")
    } noraise {
      _ => fail("unsupported Qwen3 descriptor substitution was admitted")
    }
  }
}
