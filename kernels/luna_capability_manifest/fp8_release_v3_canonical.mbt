///|
fn encode_fp8_reusable_paged_release_manifest_v3(
  identity : @spec.ModelIdentity,
  target : @catalog.DeviceTarget,
  recipe_digest : String,
  profile : @launch_contract.PagedKernelProfile,
  kv_layout_version : Int,
  tokens_per_page : Int,
  operations : ReadOnlyArray[LunaFp8ReleaseOperationV2],
  maximum : Int,
) -> Bytes raise LunaCapabilityManifestError {
  let output : Array[Byte] = []
  fp8_release_text(
    output, "lunaflux.luna-fp8-reusable-paged-release-manifest.v3", maximum,
  )
  fp8_release_text(output, identity.content().as_hex(), maximum)
  fp8_release_text(output, identity.plan().as_hex(), maximum)
  fp8_release_text(output, target.compute_major().to_string(), maximum)
  fp8_release_text(output, target.compute_minor().to_string(), maximum)
  fp8_release_text(output, recipe_digest, maximum)
  fp8_release_text(output, profile.id().as_int().to_string(), maximum)
  fp8_release_text(output, profile.max_query_rows().to_string(), maximum)
  fp8_release_text(output, profile.max_query_tokens().to_string(), maximum)
  fp8_release_text(
    output,
    profile.max_page_table_entries().to_string(),
    maximum,
  )
  fp8_release_text(output, kv_layout_version.to_string(), maximum)
  fp8_release_text(output, tokens_per_page.to_string(), maximum)
  fp8_release_text(output, operations.length().to_string(), maximum)
  for ordinal, operation in operations {
    fp8_release_text(output, ordinal.to_string(), maximum)
    fp8_release_text(
      output,
      operation.operation().as_int().to_string(),
      maximum,
    )
    fp8_release_text(
      output,
      operation.entry_point().family().family_id().as_int().to_string(),
      maximum,
    )
    fp8_release_text(
      output,
      operation.entry_point().family().module_digest().sha256(),
      maximum,
    )
    fp8_release_text(
      output,
      operation.entry_point().entry_point_id().as_int().to_string(),
      maximum,
    )
    fp8_release_text(output, operation.module_locator(), maximum)
    fp8_release_text(output, operation.function_symbol(), maximum)
    fp8_release_text(output, operation.source_digest_sha256(), maximum)
    fp8_release_text(output, operation.recipe_digest_sha256(), maximum)
    let dimensions = operation.dimensions()
    for
      value in [
        dimensions.grid_x(),
        dimensions.grid_y(),
        dimensions.grid_z(),
        dimensions.block_x(),
        dimensions.block_y(),
        dimensions.block_z(),
        dimensions.shared_memory_bytes(),
      ] {
      fp8_release_text(output, value.to_string(), maximum)
    }
    fp8_release_text(output, operation.operands().length().to_string(), maximum)
    for operand_ordinal, operand in operation.operands() {
      fp8_release_text(output, operand_ordinal.to_string(), maximum)
      fp8_release_role(output, operand.role(), maximum)
      fp8_release_text(output, operand.byte_count().to_string(), maximum)
      fp8_release_text(output, operand.alignment().to_string(), maximum)
    }
    fp8_release_text(output, operation.workspace_bytes().to_string(), maximum)
    fp8_release_text(
      output,
      operation.workspace_alignment().to_string(),
      maximum,
    )
  }
  Bytes::from_array(output)
}

///|
pub fn LunaFp8ReusablePagedReleaseManifestV3::new(
  identity~ : @spec.ModelIdentity,
  target~ : @catalog.DeviceTarget,
  reusable_recipe_digest_sha256~ : String,
  profile~ : @launch_contract.PagedKernelProfile,
  kv_layout_version~ : Int,
  tokens_per_page~ : Int,
  operations~ : ArrayView[LunaFp8ReleaseOperationV2],
  maximum_canonical_bytes~ : Int,
) -> LunaFp8ReusablePagedReleaseManifestV3 raise LunaCapabilityManifestError {
  if !luna_capability_is_sha256(reusable_recipe_digest_sha256) ||
    kv_layout_version <= 0 ||
    tokens_per_page <= 0 ||
    operations.is_empty() ||
    operations.length() > 65536 ||
    maximum_canonical_bytes <= 0 ||
    maximum_canonical_bytes > 16777216 {
    raise Invalid(Evidence)
  }
  let retained = ReadOnlyArray::from_array(operations)
  let canonical = encode_fp8_reusable_paged_release_manifest_v3(
    identity, target, reusable_recipe_digest_sha256, profile, kv_layout_version,
    tokens_per_page, retained, maximum_canonical_bytes,
  )
  {
    identity,
    target,
    reusable_recipe_digest_sha256,
    profile,
    kv_layout_version,
    tokens_per_page,
    operations: retained,
    canonical,
    digest: LunaCapabilityManifestDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ),
  }
}
