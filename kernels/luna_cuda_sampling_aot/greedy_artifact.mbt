///|
/// Admits two identical offline CUBIN builds for the exact closed lowering.
pub fn admit_greedy_sampling_cuda_artifact(
  lowering : GreedySamplingCudaLowering,
  first_module : Bytes,
  second_module : Bytes,
  receipt : GreedySamplingCompileReceipt,
  limits : GreedySamplingAotLimits,
) -> GreedySamplingCudaArtifact raise SamplingBoundaryError {
  if first_module.is_empty() || second_module.is_empty() {
    raise Invalid(ArtifactModule)
  }
  if first_module.length().to_int64() > limits.max_module_bytes ||
    second_module.length().to_int64() > limits.max_module_bytes {
    raise LimitExceeded(ArtifactModule)
  }
  if receipt.source_sha256 != lowering.source_sha256 ||
    receipt.recipe_sha256 != lowering.recipe_sha256 ||
    receipt.toolchain_sha256 !=
    lowering.compiler.identity().toolchain_digest_sha256() {
    raise Mismatch(ArtifactReceipt)
  }
  let first_sha256 = sampling_sha256(first_module)
  let second_sha256 = sampling_sha256(second_module)
  if receipt.first_module_sha256 != first_sha256 ||
    receipt.second_module_sha256 != second_sha256 {
    raise Mismatch(ArtifactReceipt)
  }
  if first_sha256 != second_sha256 || first_module != second_module {
    raise Mismatch(ArtifactDeterminism)
  }
  let module_digest = @catalog.AotArtifactDigest::from_sha256(first_sha256) catch {
    _ => raise Invalid(ArtifactDigest)
  }
  {
    target: lowering.target,
    vocabulary_size: lowering.vocabulary_size,
    maximum_rows: lowering.maximum_rows,
    result_row_bytes: GREEDY_RESULT_ROW_BYTES,
    function_symbol: lowering.function_symbol,
    module_bytes: first_module,
    module_digest,
    source_sha256: lowering.source_sha256,
    recipe_sha256: lowering.recipe_sha256,
    driver_sha256: receipt.driver_sha256,
  }
}
