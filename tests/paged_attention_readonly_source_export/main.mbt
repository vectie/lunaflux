///|
suberror ReadOnlySourceExportError {
  ReadOnlySourceExportRejected(String)
}

///|
fn require_readonly_export(condition : Bool, reason : String) -> Unit raise {
  if !condition {
    raise ReadOnlySourceExportRejected(reason)
  }
}

///|
fn readonly_export_sha256(bytes : Bytes) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn readonly_export_record(
  candidate : @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyCudaCandidate,
) -> Bytes {
  let fallback = candidate.reference_fallback()
  let target = candidate.target()
  let profile = candidate.profile()
  @utf8.encode(
    "schema=lunaflux-paged-attention-readonly-source-export.v1\n" +
    "candidate_sha256=" +
    candidate.candidate_digest().as_hex() +
    "\nsource_sha256=" +
    candidate.source_digest().as_hex() +
    "\nrecipe_sha256=" +
    candidate.recipe_digest().as_hex() +
    "\nfallback_source_sha256=" +
    fallback.source_digest().as_hex() +
    "\nfallback_recipe_sha256=" +
    fallback.recipe_digest().as_hex() +
    "\noperation_id=" +
    candidate.operation_id().as_int().to_string() +
    "\ntarget=sm_" +
    target.compute_major().to_string() +
    target.compute_minor().to_string() +
    "\nprofile_id=" +
    profile.id().as_int().to_string() +
    "\nprofile=" +
    profile.max_query_rows().to_string() +
    "," +
    profile.max_query_tokens().to_string() +
    "," +
    profile.max_page_table_entries().to_string() +
    "\nkv_layout=layer_major_split_key_value_v1\nkv_layout_stable_version=1" +
    "\nfunction_symbol=" +
    candidate.function_symbol() +
    "\nraw_pointer_abi=" +
    candidate.kernel_abi().canonical_name() +
    "\ninput_row_width=" +
    candidate.input_row_width().to_string() +
    "\ninput_activation_ref=" +
    candidate.input_activation().as_int().to_string() +
    "\ndispatch_canary_per_token=" +
    candidate.dispatch_canary_per_token().to_string() +
    "\ndispatch_canary_cell_count=" +
    candidate.profile().max_query_tokens().to_string() +
    "\nselection_precondition=standalone-positioned-rope-paged-kvwrite-complete-v1\n" +
    "kv_cache_mutation=none\nqualification_only=true\nmanifest_bindable=false\n" +
    "promotion_authority=absent\n",
  )
}

///|
async fn readonly_write_new(path : String, bytes : Bytes) -> Unit {
  @fs.write_file(
    path,
    bytes,
    create_mode=CreateNew,
    permission=0o600,
    sync=Full,
  )
}

///|
async fn readonly_export(args : Array[String]) -> Unit {
  require_readonly_export(
    args.length() == 5,
    "usage: paged_attention_readonly_source_export export ABSOLUTE_EMPTY_OUTPUT TOOLCHAIN_SHA256 13.1.115",
  )
  let output = args[2]
  require_readonly_export(
    output.has_prefix("/") && @fs.exists(output),
    "output must be an existing absolute directory",
  )
  require_readonly_export(
    args[3].length() == 64 && args[4] == "13.1.115",
    "toolchain identity is not exact",
  )
  let candidate = readonly_fixture_candidate(args[3])
  require_readonly_export(
    !candidate.manifest_bindable() && !candidate.promotion_authorized(),
    "candidate unexpectedly carries authority",
  )
  let record = readonly_export_record(candidate)
  readonly_write_new(output + "/readonly.cu", candidate.source_bytes())
  readonly_write_new(output + "/readonly.recipe", candidate.recipe_bytes())
  readonly_write_new(
    output + "/readonly.canonical",
    candidate.canonical_bytes(),
  )
  readonly_write_new(
    output + "/fallback.cu",
    candidate.reference_fallback().source_bytes(),
  )
  readonly_write_new(
    output + "/fallback.recipe",
    candidate.reference_fallback().recipe_bytes(),
  )
  readonly_write_new(output + "/export.v1", record)
  println(
    "outcome=paged-attention-readonly-source-exported export_sha256=\{readonly_export_sha256(record)} candidate_sha256=\{candidate.candidate_digest().as_hex()} authority=qualification-only",
  )
}

///|
async fn readonly_bind(args : Array[String]) -> Unit {
  require_readonly_export(
    args.length() == 8,
    "usage: paged_attention_readonly_source_export bind FIRST SECOND RECEIPT RECEIPT_SHA256 TOOLCHAIN_SHA256 13.1.115",
  )
  require_readonly_export(
    args[6].length() == 64 && args[7] == "13.1.115",
    "toolchain identity is not exact",
  )
  let candidate = readonly_fixture_candidate(args[6])
  let first = @fs.read_file(args[2]).binary()
  let second = @fs.read_file(args[3]).binary()
  let receipt = @fs.read_file(args[4]).binary()
  let binding = @luna_cuda_paged_attention_readonly_aot.bind_paged_attention_readonly_compiled_candidate(
    candidate,
    first,
    second,
    receipt,
    @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyDigest::from_sha256(
      args[5],
    ),
    @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyCompiledLimits::new(
      maximum_module_bytes=1073741824L,
      maximum_receipt_bytes=1048576,
      maximum_binding_bytes=1048576,
    ),
  )
  require_readonly_export(
    !binding.manifest_bindable() && !binding.promotion_authorized(),
    "compiled binding unexpectedly carries authority",
  )
  println(
    "outcome=paged-attention-readonly-compiled-candidate-bound binding_sha256=\{binding.binding_digest().as_hex()} module_sha256=\{binding.module_digest().as_hex()} authority=qualification-only",
  )
}

///|
async fn main {
  let args = @env.args()
  try {
    require_readonly_export(args.length() >= 2, "missing mode")
    match args[1] {
      "export" => readonly_export(args)
      "bind" => readonly_bind(args)
      _ => raise ReadOnlySourceExportRejected("mode must be export or bind")
    }
  } catch {
    ReadOnlySourceExportRejected(reason) => {
      println(
        "outcome=paged-attention-readonly-source-export-rejected reason=\{reason}",
      )
      raise ReadOnlySourceExportRejected(reason)
    }
    error => raise error
  }
}
