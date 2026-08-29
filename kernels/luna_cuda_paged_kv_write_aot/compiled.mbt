///|
pub fn positioned_paged_kv_write_compile_receipt(
  candidate : PagedKvWriteCudaCandidate,
  driver_identity_sha256 : String,
  module_bytes : Bytes,
) -> Bytes raise PagedKvWriteAotError {
  if !paged_kv_write_is_sha256(driver_identity_sha256) ||
    module_bytes.is_empty() {
    raise Invalid(CompileReceipt)
  }
  @utf8.encode(
    "schema=lunaflux-positioned-paged-kv-write-compile-receipt.v1\n" +
    "candidate_sha256=" +
    candidate.candidate_digest.as_hex() +
    "\n" +
    "source_sha256=" +
    candidate.source_digest.as_hex() +
    "\n" +
    "recipe_sha256=" +
    candidate.recipe_digest.as_hex() +
    "\n" +
    "toolchain_sha256=" +
    candidate.compiler.identity().toolchain_digest_sha256() +
    "\n" +
    "driver_identity_sha256=" +
    driver_identity_sha256 +
    "\n" +
    "target=sm_" +
    candidate.target.compute_major().to_string() +
    candidate.target.compute_minor().to_string() +
    "\n" +
    "symbol=" +
    candidate.symbol +
    "\n" +
    "abi=" +
    candidate.abi.canonical_name() +
    "\ndispatch_canary=" +
    candidate.dispatch_canary.to_string() +
    "\n" +
    "cubin_sha256=" +
    paged_kv_write_sha256(module_bytes) +
    "\n" +
    "qualification_only=true\n",
  )
}

///|
fn paged_kv_write_binding_record(
  candidate : PagedKvWriteCudaCandidate,
  module_digest : PagedKvWriteDigest,
  receipt_digest : PagedKvWriteDigest,
) -> Bytes {
  @utf8.encode(
    "schema=lunaflux-positioned-paged-kv-write-compiled-binding.v1\n" +
    "candidate_sha256=" +
    candidate.candidate_digest.as_hex() +
    "\n" +
    "source_sha256=" +
    candidate.source_digest.as_hex() +
    "\n" +
    "recipe_sha256=" +
    candidate.recipe_digest.as_hex() +
    "\n" +
    "model_content_sha256=" +
    candidate.model_content +
    "\n" +
    "model_plan_sha256=" +
    candidate.model_plan +
    "\n" +
    "attention_operation=" +
    candidate.attention_operation.as_int().to_string() +
    "\n" +
    "positioned_qkv_activation=" +
    candidate.positioned_qkv.as_int().to_string() +
    "\n" +
    "kv_layer=" +
    candidate.kv_layer.as_int().to_string() +
    "\n" +
    "tokens_per_page=" +
    candidate.tokens_per_page.to_string() +
    "\n" +
    "total_page_count=" +
    candidate.total_page_count.to_string() +
    "\n" +
    "page_stride_bytes=" +
    candidate.page_stride_bytes.to_string() +
    "\n" +
    "kv_width=" +
    candidate.kv_width.to_string() +
    "\nquery_heads=" +
    candidate.query_heads.to_string() +
    "\nkv_heads=" +
    candidate.kv_heads.to_string() +
    "\nhead_dimension=" +
    candidate.head_dimension.to_string() +
    "\n" +
    "symbol=" +
    candidate.symbol +
    "\nabi=" +
    candidate.abi.canonical_name() +
    "\ndispatch_canary_per_token=" +
    candidate.dispatch_canary.to_string() +
    "\n" +
    "fallback_source_sha256=" +
    candidate.fallback_source +
    "\n" +
    "fallback_recipe_sha256=" +
    candidate.fallback_recipe +
    "\n" +
    "cubin_sha256=" +
    module_digest.as_hex() +
    "\n" +
    "compile_receipt_sha256=" +
    receipt_digest.as_hex() +
    "\n" +
    "manifest_bindable=false\npromotion_authority=false\n",
  )
}

///|
pub fn bind_positioned_paged_kv_write_compiled_artifact(
  candidate : PagedKvWriteCudaCandidate,
  first_module : Bytes,
  second_module : Bytes,
  receipt : Bytes,
  limits : PagedKvWriteAotLimits,
) -> PagedKvWriteCompiledArtifact raise PagedKvWriteAotError {
  if first_module.is_empty() ||
    second_module.is_empty() ||
    first_module.length().to_int64() > limits.max_module_bytes ||
    second_module.length().to_int64() > limits.max_module_bytes {
    raise LimitExceeded(CompiledArtifact)
  }
  if first_module != second_module {
    raise Mismatch(Determinism)
  }
  if receipt.is_empty() || receipt.length() > limits.max_receipt_bytes {
    raise LimitExceeded(CompileReceipt)
  }
  let document = @utf8.decode(receipt) catch {
    _ => raise Invalid(CompileReceipt)
  }
  let mut driver : String? = None
  for line in document.split("\n") {
    if line.has_prefix("driver_identity_sha256=") {
      driver = Some(line[23:].to_owned())
    }
  }
  guard driver is Some(driver) && paged_kv_write_is_sha256(driver) else {
    raise Invalid(CompileReceipt)
  }
  if receipt !=
    positioned_paged_kv_write_compile_receipt(candidate, driver, first_module) {
    raise Mismatch(CompileReceipt)
  }
  let module_digest = PagedKvWriteDigest(paged_kv_write_sha256(first_module))
  let receipt_digest = PagedKvWriteDigest(paged_kv_write_sha256(receipt))
  let binding = paged_kv_write_binding_record(
    candidate, module_digest, receipt_digest,
  )
  if binding.length() > limits.max_binding_bytes {
    raise LimitExceeded(Recipe)
  }
  {
    candidate: candidate.candidate_digest,
    source: candidate.source_digest,
    recipe: candidate.recipe_digest,
    target: candidate.target,
    symbol: candidate.symbol,
    abi: candidate.abi,
    module_bytes: first_module,
    module_digest,
    receipt,
    receipt_digest,
    driver_identity: driver,
    binding,
    binding_digest: PagedKvWriteDigest(paged_kv_write_sha256(binding)),
  }
}
