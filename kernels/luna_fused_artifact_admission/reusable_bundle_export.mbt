///|
/// Canonically packages a fixed-size set of heterogeneous production modules.
/// Inputs already own validated execution metadata. Serialize once; the receiver
/// parses once. Do not hash and decode the newly serialized payload here.
pub fn export_reusable_fused_runtime_bundle(
  model_content_sha256~ : String,
  model_plan_sha256~ : String,
  target~ : @catalog.DeviceTarget,
  modules~ : ArrayView[ReusableFusedRuntimeModuleInput],
  attention_bucket_routes? : Bytes,
  vendor_projection_exact_rows~ : Int32 = 0,
) -> Bytes raise FusedArtifactAdmissionError {
  ignore(fused_deployment_digest(model_content_sha256))
  ignore(fused_deployment_digest(model_plan_sha256))
  if !target.supports_bf16() {
    raise InvalidFusedArtifact(Target)
  }
  reusable_runtime_validate_unique_inputs(modules)
  let out = StringBuilder()
  if vendor_projection_exact_rows < 0 ||
    vendor_projection_exact_rows == 1 ||
    vendor_projection_exact_rows > 1048576 {
    raise InvalidFusedArtifact(Manifest)
  }
  let partition_metadata = modules.any(m => {
    m.decode_execution is Some(descriptor) && descriptor.partition_count() != 8
  })
  let row_metadata = partition_metadata ||
    vendor_projection_exact_rows > 0 ||
    modules.any(m => m.row_variant is Some(_))
  let allow_approximate_prefill = modules.any(m => {
    m.prefill_exponential_law != StrictNaturalExponential
  })
  let prefill_metadata = row_metadata || allow_approximate_prefill
  let decode_metadata = prefill_metadata ||
    modules.any(m => {
      m.abi == PagedAttentionReadOnlyExtendedDecodeProductionV11 ||
      (match m.decode_execution {
        Some(descriptor) => descriptor.storage_geometry() != (32, 2)
        None => false
      })
    })
  let ownership = prefill_metadata ||
    modules.any(m => {
      m.heads_per_cta != 1 ||
      m.abi == AttentionIngressPreparedRotaryProductionV11 ||
      m.abi == PagedAttentionReadOnlyBlockwiseF32DecodeProductionV8 ||
      m.abi == PagedAttentionReadOnlyBlockwiseFmaF32DecodeProductionV9 ||
      m.abi.is_compiler_blockwise_decode()
    })
  let metadata = ownership ||
    attention_bucket_routes is Some(_) ||
    modules.any(m => m.query_metadata_version != 0)
  out.write_string(
    if partition_metadata {
      "schema=lunaflux-reusable-fused-runtime-bundle.v15\n"
    } else if vendor_projection_exact_rows > 0 {
      "schema=lunaflux-reusable-fused-runtime-bundle.v14\n"
    } else if row_metadata {
      "schema=lunaflux-reusable-fused-runtime-bundle.v13\n"
    } else if prefill_metadata {
      "schema=lunaflux-reusable-fused-runtime-bundle.v12\n"
    } else if decode_metadata {
      "schema=lunaflux-reusable-fused-runtime-bundle.v11\n"
    } else if modules.any(m => {
        m.abi == AttentionIngressPreparedRotaryProductionV11
      }) {
      "schema=lunaflux-reusable-fused-runtime-bundle.v10\n"
    } else if modules.any(m => {
        m.abi == PagedAttentionReadOnlyDualScoreBlockwiseF32DecodeProductionV10
      }) {
      "schema=lunaflux-reusable-fused-runtime-bundle.v9\n"
    } else if modules.any(m => {
        m.abi == PagedAttentionReadOnlyBlockwiseFmaF32DecodeProductionV9
      }) {
      "schema=lunaflux-reusable-fused-runtime-bundle.v8\n"
    } else if ownership {
      "schema=lunaflux-reusable-fused-runtime-bundle.v7\n"
    } else if attention_bucket_routes is Some(_) {
      "schema=lunaflux-reusable-fused-runtime-bundle.v6\n"
    } else if metadata {
      "schema=lunaflux-reusable-fused-runtime-bundle.v5\n"
    } else {
      "schema=lunaflux-reusable-fused-runtime-bundle.v4\n"
    },
  )
  out.write_string("model_content_sha256=" + model_content_sha256 + "\n")
  out.write_string("model_plan_sha256=" + model_plan_sha256 + "\n")
  reusable_runtime_append_target(out, target)
  if prefill_metadata {
    out.write_string(
      "allow_approximate_prefill_exponential=" +
      (if allow_approximate_prefill { "1" } else { "0" }) +
      "\n",
    )
  }
  if attention_bucket_routes is Some(routes) {
    if routes.is_empty() || routes.length() > 1048576 {
      raise InvalidFusedArtifact(Manifest)
    }
    out.write_string(
      "attention_bucket_routes_hex=" +
      @crypto.bytes_to_hex_string(routes) +
      "\n",
    )
  } else if ownership {
    out.write_string("attention_bucket_routes_hex=-\n")
  }
  if vendor_projection_exact_rows > 0 || partition_metadata {
    out.write_string(
      "vendor_projection_exact_rows=" +
      vendor_projection_exact_rows.to_string() +
      "\n",
    )
  }
  out.write_string("module_count=" + modules.length().to_string() + "\n")
  let canonical_abis = [
    ResidualRmsNormProductionV2,
    QwenQkvQkNormPositionedRopePagedKvWriteProductionV4,
    QwenQkNormPositionedRopePagedKvWriteProductionV2,
    QwenPagedAttentionPrefillPartitionedPartialProductionV4,
    QwenPagedAttentionPrefillPartitionedMergeProductionV4,
    QwenPagedAttentionPrefillPartitioned8PartialProductionV6,
    QwenPagedAttentionPrefillPartitioned8MergeProductionV6,
    QwenPagedAttentionPrefillWideQueryProductionV5,
    QwenPagedAttentionReadOnlyProductionV2,
    QwenPagedAttentionReadOnlyDecodeSplitProductionV3,
    PagedAttentionReadOnlyCompilerDecodeProductionV7,
    PagedAttentionReadOnlyBlockwiseF32DecodeProductionV8,
    PagedAttentionReadOnlyBlockwiseFmaF32DecodeProductionV9,
    PagedAttentionReadOnlyDualScoreBlockwiseF32DecodeProductionV10,
    AttentionIngressPreparedRotaryProductionV11,
    PagedAttentionReadOnlyExtendedDecodeProductionV11,
  ]
  let mut ordinal = 0
  for abi in canonical_abis {
    for module_ in modules {
      if module_.abi == abi {
        reusable_runtime_append_module(
          out, ordinal, module_, metadata, ownership, decode_metadata, prefill_metadata,
          row_metadata, partition_metadata,
        )
        ordinal += 1
      }
    }
  }
  @utf8.encode(out.to_string())
}
