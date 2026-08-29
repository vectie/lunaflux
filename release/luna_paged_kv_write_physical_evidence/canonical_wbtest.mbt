///|
fn kv_fixture_line(
  output : StringBuilder,
  key : String,
  value : String,
) -> Unit {
  output.write_string(key + "=" + value + "\n")
}

///|
fn kv_physical_fixture_canonical(fixture : KvPhysicalFixture) -> Bytes {
  let candidate = fixture.candidate
  let compiled = fixture.compiled
  let output = StringBuilder()
  kv_fixture_line(
    output, "schema", "lunaflux-paged-kv-write-physical-evidence.v1",
  )
  kv_fixture_line(output, "approved_policy_sha256", "9".repeat(64))
  for
    pair in [
      ("device_uuid", "GPU-test-sm120"),
      ("device_pci", "00000000:01:00.0"),
      ("host_driver_version", "590.48.01"),
    ] {
    kv_fixture_line(output, pair.0, pair.1)
  }
  kv_fixture_line(output, "device_name_sha256", "1".repeat(64))
  kv_fixture_line(output, "device_total_memory_bytes", "17179869184")
  kv_fixture_line(output, "device_compute", "12.0")
  kv_fixture_line(output, "cuda_driver_version", "13010")
  kv_fixture_line(output, "context_device_uuid", "GPU-test-sm120")
  kv_fixture_line(output, "context_device_pci", "00000000:01:00.0")
  for
    key in [
      "nvcc_sha256", "ptxas_sha256", "cuobjdump_sha256", "nvdisasm_sha256", "compute_sanitizer_sha256",
      "nvidia_smi_sha256", "driver_record_sha256", "nvcc_version_sha256", "ptxas_version_sha256",
      "cuobjdump_version_sha256", "nvdisasm_version_sha256", "compute_sanitizer_version_sha256",
    ] {
    kv_fixture_line(output, key, "2".repeat(64))
  }
  kv_fixture_line(
    output,
    "driver_identity_sha256",
    compiled.driver_identity_sha256(),
  )
  for
    pair in [
      ("candidate_sha256", candidate.candidate_digest().as_hex()),
      ("source_sha256", candidate.source_digest().as_hex()),
      ("recipe_sha256", candidate.recipe_digest().as_hex()),
      ("compiled_binding_sha256", compiled.binding_digest().as_hex()),
      ("module_sha256", compiled.module_digest().as_hex()),
      ("compile_receipt_sha256", compiled.compile_receipt_digest().as_hex()),
      ("serial_oracle_source_sha256", "b".repeat(64)),
      ("scalar_referee_source_sha256", "c".repeat(64)),
      ("probe_executable_sha256", "e".repeat(64)),
      ("model_content_sha256", candidate.model_content_digest_sha256()),
      ("model_plan_sha256", candidate.model_plan_digest_sha256()),
    ] {
    kv_fixture_line(output, pair.0, pair.1)
  }
  kv_fixture_line(
    output,
    "attention_operation",
    candidate.operation_id().as_int().to_string(),
  )
  kv_fixture_line(
    output,
    "positioned_qkv_activation",
    candidate.positioned_qkv_activation().as_int().to_string(),
  )
  kv_fixture_line(output, "query_heads", candidate.query_heads().to_string())
  kv_fixture_line(output, "kv_heads", candidate.kv_heads().to_string())
  kv_fixture_line(
    output,
    "head_dimension",
    candidate.head_dimension().to_string(),
  )
  kv_fixture_line(output, "positioned_qkv_row_width", "256")
  kv_fixture_line(output, "target", "sm_120")
  kv_fixture_line(output, "max_query_rows", "4")
  kv_fixture_line(output, "max_query_tokens", "8")
  kv_fixture_line(output, "max_page_table_entries", "16")
  kv_fixture_line(
    output,
    "tokens_per_page",
    candidate.tokens_per_page().to_string(),
  )
  kv_fixture_line(
    output,
    "total_page_count",
    candidate.total_page_count().to_string(),
  )
  kv_fixture_line(
    output,
    "page_stride_bytes",
    candidate.page_stride_bytes().to_string(),
  )
  kv_fixture_line(output, "toolchain_sha256", "9".repeat(64))
  kv_fixture_line(output, "compiler_version", "13.1.115")
  kv_fixture_line(output, "function_symbol", candidate.function_symbol())
  kv_fixture_line(
    output,
    "raw_pointer_abi",
    candidate.kernel_abi().canonical_name(),
  )
  let dimensions = candidate.launch_dimensions()
  for
    pair in [
      ("grid_x", dimensions.grid_x()),
      ("grid_y", dimensions.grid_y()),
      ("grid_z", dimensions.grid_z()),
      ("block_x", dimensions.block_x()),
      ("block_y", dimensions.block_y()),
      ("block_z", dimensions.block_z()),
      ("shared_memory_bytes", dimensions.shared_memory_bytes()),
    ] {
    kv_fixture_line(output, pair.0, pair.1.to_string())
  }
  kv_fixture_line(
    output,
    "dispatch_canary_per_token",
    candidate.dispatch_canary_per_token().to_string(),
  )
  kv_fixture_line(
    output,
    "fallback_source_sha256",
    candidate.fallback_source_digest_sha256(),
  )
  kv_fixture_line(
    output,
    "fallback_recipe_sha256",
    candidate.fallback_recipe_digest_sha256(),
  )
  kv_fixture_line(output, "case_count", "6")
  for
    key in [
      "prefill_case_pass", "decode_case_pass", "mixed_case_pass", "origin_case_pass",
      "page_tail_case_pass", "cross_page_case_pass", "multirow_case_pass", "full_grid_launch_pass",
      "expected_cells_pass", "byte_oracle_pass", "scalar_oracle_pass", "serial_cuda_oracle_pass",
      "guards_unchanged", "non_target_cache_unchanged", "dispatch_canary_exact",
      "deterministic_cubins", "memcheck_status", "racecheck_status", "initcheck_status",
    ] {
    kv_fixture_line(output, key, "pass")
  }
  kv_fixture_line(output, "qualified_token_count", "8")
  kv_fixture_line(output, "kv_width", candidate.kv_width().to_string())
  for
    key in [
      "expected_key_mutations", "observed_key_mutations", "expected_value_mutations",
      "observed_value_mutations",
    ] {
    kv_fixture_line(output, key, "512")
  }
  kv_fixture_line(output, "dispatch_canary_cell_count", "8")
  kv_fixture_line(output, "expected_dispatch_canary", "178568")
  kv_fixture_line(output, "observed_dispatch_canary", "178568")
  kv_fixture_line(output, "cuobjdump_sass_sha256", "3".repeat(64))
  kv_fixture_line(output, "cuobjdump_sass_instruction_count", "17")
  kv_fixture_line(output, "nvdisasm_sass_sha256", "4".repeat(64))
  kv_fixture_line(output, "nvdisasm_sass_instruction_count", "17")
  kv_fixture_line(output, "sass_global_load_observed", "true")
  kv_fixture_line(output, "sass_global_store_observed", "true")
  kv_fixture_line(output, "registers_per_thread", "64")
  for
    key in [
      "static_shared_bytes", "local_bytes", "stack_bytes", "spill_store_bytes", "spill_load_bytes",
      "contexts_live_after", "streams_live_after", "allocations_live_after", "modules_live_after",
      "functions_live_after", "device_bytes_live_after", "pending_work_after", "stderr_bytes",
    ] {
    kv_fixture_line(output, key, "0")
  }
  kv_fixture_line(output, "physical_cuda_observed", "true")
  kv_fixture_line(output, "synthetic_test_only", "false")
  kv_fixture_line(output, "files_manifest_sha256", "5".repeat(64))
  kv_fixture_line(output, "outer_seal_sha256", "6".repeat(64))
  kv_fixture_line(output, "manifest_bindable", "false")
  kv_fixture_line(output, "promotion_authority", "absent")
  kv_fixture_line(output, "runtime_dispatch_authority", "absent")
  @utf8.encode(output.to_string())
}
