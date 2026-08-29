///|
pub struct PagedAttentionReadOnlyCompiledLimits {
  priv maximum_module_bytes : Int64
  priv maximum_receipt_bytes : Int
  priv maximum_binding_bytes : Int
}

///|
pub fn PagedAttentionReadOnlyCompiledLimits::new(
  maximum_module_bytes~ : Int64,
  maximum_receipt_bytes~ : Int,
  maximum_binding_bytes~ : Int,
) -> PagedAttentionReadOnlyCompiledLimits raise PagedAttentionReadOnlyAotError {
  if maximum_module_bytes <= 0L ||
    maximum_module_bytes > 1073741824L ||
    maximum_receipt_bytes <= 0 ||
    maximum_receipt_bytes > 1048576 ||
    maximum_binding_bytes <= 0 ||
    maximum_binding_bytes > 1048576 {
    raise PagedAttentionReadOnlyInvalid(Limits)
  }
  { maximum_module_bytes, maximum_receipt_bytes, maximum_binding_bytes }
}

///|
pub struct PagedAttentionReadOnlyCompiledBinding {
  priv candidate_digest : PagedAttentionReadOnlyDigest
  priv source_digest : PagedAttentionReadOnlyDigest
  priv recipe_digest : PagedAttentionReadOnlyDigest
  priv fallback_source_digest : String
  priv module_bytes : Bytes
  priv module_digest : PagedAttentionReadOnlyDigest
  priv compile_receipt : Bytes
  priv compile_receipt_digest : PagedAttentionReadOnlyDigest
  priv binding_record : Bytes
  priv binding_digest : PagedAttentionReadOnlyDigest
}

///|
pub fn PagedAttentionReadOnlyCompiledBinding::candidate_digest(
  self : PagedAttentionReadOnlyCompiledBinding,
) -> PagedAttentionReadOnlyDigest {
  self.candidate_digest
}

///|
pub fn PagedAttentionReadOnlyCompiledBinding::source_digest(
  self : PagedAttentionReadOnlyCompiledBinding,
) -> PagedAttentionReadOnlyDigest {
  self.source_digest
}

///|
pub fn PagedAttentionReadOnlyCompiledBinding::recipe_digest(
  self : PagedAttentionReadOnlyCompiledBinding,
) -> PagedAttentionReadOnlyDigest {
  self.recipe_digest
}

///|
pub fn PagedAttentionReadOnlyCompiledBinding::fallback_source_digest(
  self : PagedAttentionReadOnlyCompiledBinding,
) -> String {
  self.fallback_source_digest
}

///|
pub fn PagedAttentionReadOnlyCompiledBinding::module_bytes(
  self : PagedAttentionReadOnlyCompiledBinding,
) -> Bytes {
  self.module_bytes
}

///|
pub fn PagedAttentionReadOnlyCompiledBinding::module_digest(
  self : PagedAttentionReadOnlyCompiledBinding,
) -> PagedAttentionReadOnlyDigest {
  self.module_digest
}

///|
pub fn PagedAttentionReadOnlyCompiledBinding::compile_receipt_bytes(
  self : PagedAttentionReadOnlyCompiledBinding,
) -> Bytes {
  self.compile_receipt
}

///|
pub fn PagedAttentionReadOnlyCompiledBinding::compile_receipt_digest(
  self : PagedAttentionReadOnlyCompiledBinding,
) -> PagedAttentionReadOnlyDigest {
  self.compile_receipt_digest
}

///|
pub fn PagedAttentionReadOnlyCompiledBinding::binding_record_bytes(
  self : PagedAttentionReadOnlyCompiledBinding,
) -> Bytes {
  self.binding_record
}

///|
pub fn PagedAttentionReadOnlyCompiledBinding::binding_digest(
  self : PagedAttentionReadOnlyCompiledBinding,
) -> PagedAttentionReadOnlyDigest {
  self.binding_digest
}

///|
pub fn PagedAttentionReadOnlyCompiledBinding::manifest_bindable(
  _self : PagedAttentionReadOnlyCompiledBinding,
) -> Bool {
  false
}

///|
pub fn PagedAttentionReadOnlyCompiledBinding::promotion_authorized(
  _self : PagedAttentionReadOnlyCompiledBinding,
) -> Bool {
  false
}

///|
fn readonly_receipt_value(
  lines : ArrayView[String],
  index : Int,
  key : String,
) -> String raise PagedAttentionReadOnlyAotError {
  if index < 0 || index >= lines.length() {
    raise PagedAttentionReadOnlyInvalid(CompileReceipt)
  }
  let prefix = key + "="
  if !lines[index].has_prefix(prefix) ||
    lines[index].length() == prefix.length() {
    raise PagedAttentionReadOnlyInvalid(CompileReceipt)
  }
  lines[index][prefix.length():].to_owned()
}

///|
pub fn bind_paged_attention_readonly_compiled_candidate(
  candidate : PagedAttentionReadOnlyCudaCandidate,
  first_module : Bytes,
  second_module : Bytes,
  compile_receipt : Bytes,
  expected_receipt_digest : PagedAttentionReadOnlyDigest,
  limits : PagedAttentionReadOnlyCompiledLimits,
) -> PagedAttentionReadOnlyCompiledBinding raise PagedAttentionReadOnlyAotError {
  if first_module.length() == 0 ||
    second_module.length() == 0 ||
    first_module.length().to_int64() > limits.maximum_module_bytes ||
    second_module.length().to_int64() > limits.maximum_module_bytes {
    raise PagedAttentionReadOnlyLimitExceeded(CompiledArtifact)
  }
  if compile_receipt.length() == 0 ||
    compile_receipt.length() > limits.maximum_receipt_bytes {
    raise PagedAttentionReadOnlyLimitExceeded(CompileReceipt)
  }
  if readonly_digest(candidate.source) != candidate.source_digest ||
    readonly_digest(candidate.recipe) != candidate.recipe_digest ||
    readonly_digest(candidate.canonical) != candidate.candidate_digest {
    raise PagedAttentionReadOnlyMismatch(Digest)
  }
  if first_module != second_module {
    raise PagedAttentionReadOnlyMismatch(Determinism)
  }
  let first_digest = readonly_digest(first_module)
  if readonly_digest(second_module) != first_digest {
    raise PagedAttentionReadOnlyMismatch(Determinism)
  }
  let receipt_digest = readonly_digest(compile_receipt)
  if receipt_digest != expected_receipt_digest {
    raise PagedAttentionReadOnlyMismatch(CompileReceipt)
  }
  let receipt_document = @utf8.decode(compile_receipt) catch {
    _ => raise PagedAttentionReadOnlyInvalid(CompileReceipt)
  }
  let lines = receipt_document
    .split("\n")
    .map(line => line.to_owned())
    .to_array()
  if lines.length() != 21 ||
    lines[20] != "" ||
    lines[0] != "schema=lunaflux-paged-attention-readonly-aot-evidence.v1" {
    raise PagedAttentionReadOnlyInvalid(CompileReceipt)
  }
  let driver_identity = readonly_receipt_value(
    lines, 7, "driver_identity_sha256",
  )
  if !readonly_is_sha256(driver_identity) {
    raise PagedAttentionReadOnlyInvalid(CompileReceipt)
  }
  let target = candidate.reference.target()
  let expected = @utf8.encode(
    "schema=lunaflux-paged-attention-readonly-aot-evidence.v1\n" +
    "candidate_sha256=" +
    candidate.candidate_digest.as_hex() +
    "\nsource_sha256=" +
    candidate.source_digest.as_hex() +
    "\nrecipe_sha256=" +
    candidate.recipe_digest.as_hex() +
    "\nfallback_source_sha256=" +
    candidate.reference.source_digest().as_hex() +
    "\nfallback_recipe_sha256=" +
    candidate.reference.recipe_digest().as_hex() +
    "\ntoolchain_sha256=" +
    candidate.reference.compiler_policy().identity().toolchain_digest_sha256() +
    "\ndriver_identity_sha256=" +
    driver_identity +
    "\ntarget=sm_" +
    target.compute_major().to_string() +
    target.compute_minor().to_string() +
    "\nfunction_symbol=" +
    candidate.function_symbol +
    "\nraw_pointer_abi=" +
    candidate.kernel_abi.canonical_name() +
    "\ninput_row_width=" +
    candidate.input_row_width.to_string() +
    "\ninput_activation_ref=" +
    candidate.input_activation.as_int().to_string() +
    "\ndispatch_canary_per_token=" +
    candidate.dispatch_canary_per_token.to_string() +
    "\ndispatch_canary_cell_count=" +
    candidate.profile().max_query_tokens().to_string() +
    "\nartifact_sha256=" +
    first_digest.as_hex() +
    "\nfirst_build_sha256=" +
    first_digest.as_hex() +
    "\nsecond_build_sha256=" +
    first_digest.as_hex() +
    "\ndeterministic=1\nkv_cache_mutation=none\n",
  )
  if compile_receipt != expected {
    raise PagedAttentionReadOnlyMismatch(CompileReceipt)
  }
  let binding_record = @utf8.encode(
    "schema=lunaflux-paged-attention-readonly-compiled-binding.v1\n" +
    "candidate_sha256=" +
    candidate.candidate_digest.as_hex() +
    "\nsource_sha256=" +
    candidate.source_digest.as_hex() +
    "\nrecipe_sha256=" +
    candidate.recipe_digest.as_hex() +
    "\nfallback_source_sha256=" +
    candidate.reference.source_digest().as_hex() +
    "\nfallback_recipe_sha256=" +
    candidate.reference.recipe_digest().as_hex() +
    "\nartifact_sha256=" +
    first_digest.as_hex() +
    "\ncompile_receipt_sha256=" +
    receipt_digest.as_hex() +
    "\nfunction_symbol=" +
    candidate.function_symbol +
    "\nraw_pointer_abi=" +
    candidate.kernel_abi.canonical_name() +
    "\ninput_row_width=" +
    candidate.input_row_width.to_string() +
    "\ninput_activation_ref=" +
    candidate.input_activation.as_int().to_string() +
    "\ndispatch_canary_per_token=" +
    candidate.dispatch_canary_per_token.to_string() +
    "\ndispatch_canary_cell_count=" +
    candidate.profile().max_query_tokens().to_string() +
    "\nselection_precondition=standalone-positioned-rope-paged-kvwrite-complete-v1\n" +
    "kv_cache_mutation=none\ncandidate_only=true\ncompiler_authority=absent\n" +
    "deployment_approval=absent\npromotion_authority=absent\nmanifest_bindable=false\n",
  )
  if binding_record.length() > limits.maximum_binding_bytes {
    raise PagedAttentionReadOnlyLimitExceeded(Evidence)
  }
  {
    candidate_digest: candidate.candidate_digest,
    source_digest: candidate.source_digest,
    recipe_digest: candidate.recipe_digest,
    fallback_source_digest: candidate.reference.source_digest().as_hex(),
    module_bytes: first_module,
    module_digest: first_digest,
    compile_receipt,
    compile_receipt_digest: receipt_digest,
    binding_record,
    binding_digest: readonly_digest(binding_record),
  }
}
