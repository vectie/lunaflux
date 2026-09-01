///|
pub fn qknorm_rope_paged_kv_write_compile_receipt(
  candidate : QkNormRopePagedKvWriteCudaCandidate,
  driver_identity_sha256 : String,
  module_bytes : Bytes,
) -> Bytes raise QkNormRopePagedKvWriteAotError {
  if !postprocess_is_sha256(driver_identity_sha256) || module_bytes.is_empty() {
    raise Invalid(CompileReceipt)
  }
  @utf8.encode(
    "schema=lunaflux-qknorm-rope-paged-kv-write-compile-receipt.v1\n" +
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
    "\n" +
    "cubin_sha256=" +
    postprocess_sha256(module_bytes) +
    "\n" +
    "reproducible_double_compile=true\n",
  )
}

///|
pub fn admit_reproducibly_compiled_qknorm_rope_paged_kv_write_module(
  candidate : QkNormRopePagedKvWriteCudaCandidate,
  first_module : Bytes,
  second_module : Bytes,
  receipt : Bytes,
  limits : QkNormRopePagedKvWriteAotLimits,
) -> QkNormRopePagedKvWriteCompiledModule raise QkNormRopePagedKvWriteAotError {
  if first_module.is_empty() ||
    second_module.is_empty() ||
    first_module.length().to_int64() > limits.max_module_bytes ||
    second_module.length().to_int64() > limits.max_module_bytes {
    raise LimitExceeded(CompiledModule)
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
  guard driver is Some(driver) && postprocess_is_sha256(driver) else {
    raise Invalid(CompileReceipt)
  }
  if receipt !=
    qknorm_rope_paged_kv_write_compile_receipt(candidate, driver, first_module) {
    raise Mismatch(CompileReceipt)
  }
  {
    candidate_digest: candidate.candidate_digest,
    module_bytes: first_module,
    module_digest: QkNormRopePagedKvWriteDigest(
      postprocess_sha256(first_module),
    ),
    receipt,
    receipt_digest: QkNormRopePagedKvWriteDigest(postprocess_sha256(receipt)),
    driver_identity: driver,
  }
}

///|
fn postprocess_binding_record(
  candidate : QkNormRopePagedKvWriteCudaCandidate,
  compiled : QkNormRopePagedKvWriteCompiledModule,
) -> Bytes {
  let (_, norm_operation, rope_operation, attention_operation) = candidate.semantic.operation_indices()
  @utf8.encode(
    "schema=lunaflux-qknorm-rope-paged-kv-write-compiled-binding.v1\n" +
    "candidate_sha256=" +
    candidate.candidate_digest.as_hex() +
    "\n" +
    "source_sha256=" +
    candidate.source_digest.as_hex() +
    "\n" +
    "recipe_sha256=" +
    candidate.recipe_digest.as_hex() +
    "\n" +
    "cubin_sha256=" +
    compiled.module_digest.as_hex() +
    "\n" +
    "compile_receipt_sha256=" +
    compiled.receipt_digest.as_hex() +
    "\n" +
    "driver_identity_sha256=" +
    compiled.driver_identity +
    "\n" +
    "model_content_sha256=" +
    candidate.model_content +
    "\n" +
    "model_plan_sha256=" +
    candidate.model_plan +
    "\n" +
    "operation_span=" +
    norm_operation.to_string() +
    "," +
    rope_operation.to_string() +
    "," +
    attention_operation.to_string() +
    "\n" +
    "symbol=" +
    candidate.symbol +
    "\n" +
    "abi=" +
    candidate.abi.canonical_name() +
    "\n" +
    "alias_contract=" +
    candidate.alias_contract.canonical_name() +
    "\n" +
    "manifest_bindable=false\npromotion_authority=false\n" +
    "physical_qualification=required\nexternal_approval=required\n",
  )
}

///|
pub fn bind_compiled_qknorm_rope_paged_kv_write_cuda_aot(
  candidate : QkNormRopePagedKvWriteCudaCandidate,
  compiled : QkNormRopePagedKvWriteCompiledModule,
  limits : QkNormRopePagedKvWriteAotLimits,
) -> QkNormRopePagedKvWriteCudaBinding raise QkNormRopePagedKvWriteAotError {
  if compiled.candidate_digest != candidate.candidate_digest {
    raise Mismatch(Binding)
  }
  let canonical = postprocess_binding_record(candidate, compiled)
  if canonical.length() > limits.max_binding_bytes {
    raise LimitExceeded(Binding)
  }
  {
    candidate,
    compiled,
    canonical,
    digest: QkNormRopePagedKvWriteDigest(postprocess_sha256(canonical)),
  }
}
