///|
async fn kv_write_new(path : String, bytes : Bytes) -> Unit {
  @fs.write_file(
    path,
    bytes,
    create_mode=CreateNew,
    permission=0o600,
    sync=Full,
  )
}

///|
fn kv_export_record(
  candidate : @luna_cuda_paged_kv_write_aot.PagedKvWriteCudaCandidate,
  oracle : Bytes,
) -> Bytes {
  let compiler = candidate.compiler_policy()
  let dimensions = candidate.launch_dimensions()
  let output = StringBuilder()
  for
    pair in [
      ("schema", "lunaflux-paged-kv-write-probe-export.v1"),
      ("candidate_sha256", candidate.candidate_digest().as_hex()),
      ("source_sha256", candidate.source_digest().as_hex()),
      ("recipe_sha256", candidate.recipe_digest().as_hex()),
      (
        "serial_oracle_source_sha256",
        @crypto.bytes_to_hex_string(@crypto.sha256(oracle)),
      ),
      ("model_content_sha256", candidate.model_content_digest_sha256()),
      ("model_plan_sha256", candidate.model_plan_digest_sha256()),
      ("attention_operation", candidate.operation_id().as_int().to_string()),
      (
        "positioned_qkv_activation",
        candidate.positioned_qkv_activation().as_int().to_string(),
      ),
      ("query_heads", candidate.query_heads().to_string()),
      ("kv_heads", candidate.kv_heads().to_string()),
      ("head_dimension", candidate.head_dimension().to_string()),
      ("positioned_qkv_row_width", "256"),
      ("target", "sm_120"),
      ("max_query_rows", candidate.profile().max_query_rows().to_string()),
      ("max_query_tokens", candidate.profile().max_query_tokens().to_string()),
      (
        "max_page_table_entries",
        candidate.profile().max_page_table_entries().to_string(),
      ),
      ("tokens_per_page", candidate.tokens_per_page().to_string()),
      ("total_page_count", candidate.total_page_count().to_string()),
      ("page_stride_bytes", candidate.page_stride_bytes().to_string()),
      ("toolchain_sha256", compiler.identity().toolchain_digest_sha256()),
      ("compiler_version", "13.1.115"),
      ("function_symbol", candidate.function_symbol()),
      ("raw_pointer_abi", candidate.kernel_abi().canonical_name()),
      ("grid_x", dimensions.grid_x().to_string()),
      ("grid_y", dimensions.grid_y().to_string()),
      ("grid_z", dimensions.grid_z().to_string()),
      ("block_x", dimensions.block_x().to_string()),
      ("block_y", dimensions.block_y().to_string()),
      ("block_z", dimensions.block_z().to_string()),
      ("shared_memory_bytes", dimensions.shared_memory_bytes().to_string()),
      (
        "dispatch_canary_per_token",
        candidate.dispatch_canary_per_token().to_string(),
      ),
      ("fallback_source_sha256", candidate.fallback_source_digest_sha256()),
      ("fallback_recipe_sha256", candidate.fallback_recipe_digest_sha256()),
      ("qualification_only", "true"),
      ("manifest_bindable", "false"),
      ("promotion_authority", "absent"),
    ] {
    output.write_string(pair.0 + "=" + pair.1 + "\n")
  }
  @utf8.encode(output.to_string())
}

///|
async fn kv_export(args : Array[String]) -> Unit {
  require_kv_probe(
    args.length() == 6,
    "usage: export TOOLCHAIN_SHA256 SOURCE_PATH RECIPE_PATH ORACLE_SOURCE_PATH",
  )
  let candidate = kv_probe_candidate(args[2])
  let oracle = kv_serial_oracle_source()
  kv_write_new(args[3], candidate.source_bytes())
  kv_write_new(args[4], candidate.recipe_bytes())
  kv_write_new(args[5], oracle)
  // The export record is adjacent to the recipe by a fixed suffix so callers
  // cannot redirect an extra authority-bearing path through this mode.
  let record_path = args[4] + ".export.v1"
  kv_write_new(record_path, kv_export_record(candidate, oracle))
  println(
    "outcome=paged-kv-write-source-exported candidate_sha256=\{candidate.candidate_digest().as_hex()} authority=qualification-only",
  )
}
