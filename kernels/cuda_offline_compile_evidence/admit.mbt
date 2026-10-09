///|
fn valid_compile_sha256(value : StringView) -> Bool {
  value.length() == 64 &&
  value.all(character => {
    character.is_ascii_digit() || (character >= 'a' && character <= 'f')
  })
}

///|
fn valid_compile_symbol(value : StringView, maximum : Int) -> Bool {
  if value.is_empty() || value.length() > maximum {
    return false
  }
  for index, character in value {
    let initial = character.is_ascii_alphabetic() || character == '_'
    if (index == 0 && !initial) ||
      (index > 0 && !initial && !character.is_ascii_digit()) {
      return false
    }
  }
  true
}

///|
fn compile_bool(value : Bool) -> String {
  if value {
    "1"
  } else {
    "0"
  }
}

///|
fn expected_compile_receipt(
  source_digest_sha256 : String,
  recipe_digest_sha256 : String,
  compiler : @luna_specialization.LunaCompilerPolicy,
  driver_identity_sha256 : String,
  target : @catalog.DeviceTarget,
  function_symbol : String,
  module_digest_sha256 : String,
) -> Bytes {
  let identity = compiler.identity()
  @utf8.encode(
    "schema=lunaflux-cuda-offline-compile-evidence.v1\n" +
    "source_sha256=" +
    source_digest_sha256 +
    "\nrecipe_sha256=" +
    recipe_digest_sha256 +
    "\ntoolchain_sha256=" +
    identity.toolchain_digest_sha256() +
    "\ncompiler_major=" +
    identity.major().to_string() +
    "\ncompiler_minor=" +
    identity.minor().to_string() +
    "\ncompiler_patch=" +
    identity.patch().to_string() +
    "\noptimization_level=" +
    compiler.optimization_level().to_string() +
    "\nfused_multiply_add=" +
    compile_bool(compiler.fused_multiply_add()) +
    "\nreassociate=" +
    compile_bool(compiler.reassociate()) +
    "\nmax_registers=" +
    compiler.max_registers().to_string() +
    "\ndriver_identity_sha256=" +
    driver_identity_sha256 +
    "\ntarget_major=" +
    target.compute_major().to_string() +
    "\ntarget_minor=" +
    target.compute_minor().to_string() +
    "\ntarget_bf16=" +
    compile_bool(target.supports_bf16()) +
    "\ntarget_cublas_lt=" +
    compile_bool(target.supports_cublas_lt()) +
    "\nfunction_symbol=" +
    function_symbol +
    "\nmodule_sha256=" +
    module_digest_sha256 +
    "\nfirst_build_sha256=" +
    module_digest_sha256 +
    "\nsecond_build_sha256=" +
    module_digest_sha256 +
    "\ndeterministic=1\n",
  )
}

///|
/// Authenticates the exact self-consistency of an offline compile receipt and
/// two byte-identical module snapshots. This performs no compilation and does
/// not authenticate who produced the receipt.
pub fn admit_deterministic_cuda_output(
  source_digest_sha256~ : String,
  recipe_digest_sha256~ : String,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  driver_identity_sha256~ : String,
  target~ : @catalog.DeviceTarget,
  function_symbol~ : String,
  first_module~ : Bytes,
  second_module~ : Bytes,
  receipt_bytes~ : Bytes,
  expected_receipt_digest_sha256~ : String,
  limits~ : CudaOfflineCompileEvidenceLimits,
) -> CudaOfflineCompileEvidence raise CudaOfflineCompileEvidenceError {
  for
    entry in [
      (source_digest_sha256, Source),
      (recipe_digest_sha256, Recipe),
      (driver_identity_sha256, Driver),
      (expected_receipt_digest_sha256, Receipt),
    ] {
    let (value, kind) = entry
    if !valid_compile_sha256(value) {
      raise InvalidDigest(kind)
    }
  }
  if !valid_compile_symbol(function_symbol, limits.maximum_symbol_bytes) {
    raise InvalidFunctionSymbol
  }
  if first_module.is_empty() || second_module.is_empty() {
    raise EmptyModule
  }
  if first_module.length().to_int64() > limits.maximum_module_bytes ||
    second_module.length().to_int64() > limits.maximum_module_bytes {
    raise ModuleTooLarge
  }
  let first_digest = @crypto.bytes_to_hex_string(@crypto.sha256(first_module))
  let second_digest = @crypto.bytes_to_hex_string(@crypto.sha256(second_module))
  if first_digest != second_digest || first_module != second_module {
    raise NondeterministicModule
  }
  if receipt_bytes.is_empty() {
    raise EmptyReceipt
  }
  if receipt_bytes.length() > limits.maximum_receipt_bytes {
    raise ReceiptTooLarge
  }
  let receipt_digest = @crypto.bytes_to_hex_string(
    @crypto.sha256(receipt_bytes),
  )
  if receipt_digest != expected_receipt_digest_sha256 {
    raise ReceiptDigestMismatch
  }
  let expected = expected_compile_receipt(
    source_digest_sha256, recipe_digest_sha256, compiler, driver_identity_sha256,
    target, function_symbol, first_digest,
  )
  if receipt_bytes != expected {
    raise ReceiptContentMismatch
  }
  let module_digest = @catalog.AotArtifactDigest::from_sha256(first_digest) catch {
    _ => raise InvalidModuleDigest
  }
  {
    source_digest_sha256,
    recipe_digest_sha256,
    compiler,
    driver_identity_sha256,
    target,
    function_symbol,
    module_digest,
    module_byte_count: first_module.length(),
    receipt_digest: CudaOfflineCompileEvidenceDigest(receipt_digest),
  }
}
