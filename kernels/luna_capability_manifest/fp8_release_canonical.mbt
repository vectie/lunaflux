///|
fn fp8_release_require(
  output : Array[Byte],
  additional : Int,
  maximum : Int,
) -> Unit raise LunaCapabilityManifestError {
  if additional < 0 || output.length() > maximum - additional {
    raise Invalid(CanonicalBytes)
  }
}

///|
fn fp8_release_text(
  output : Array[Byte],
  value : String,
  maximum : Int,
) -> Unit raise LunaCapabilityManifestError {
  let bytes = @utf8.encode(value)
  fp8_release_require(output, 8, maximum)
  for byte in bytes.length().to_int64().reinterpret_as_uint64().to_le_bytes() {
    output.push(byte)
  }
  fp8_release_require(output, bytes.length(), maximum)
  for byte in bytes {
    output.push(byte)
  }
}

///|
fn fp8_release_role(
  output : Array[Byte],
  role : @launch_contract.AotOperandRole,
  maximum : Int,
) -> Unit raise LunaCapabilityManifestError {
  match role {
    ExternalInput(TokenIds) =>
      fp8_release_text(output, "external:token-ids", maximum)
    WeightInput(reference) => {
      fp8_release_text(output, "weight", maximum)
      fp8_release_text(output, reference.as_int().to_string(), maximum)
    }
    WeightScaleInput(weight, scale) => {
      fp8_release_text(output, "weight-scale", maximum)
      fp8_release_text(output, weight.as_int().to_string(), maximum)
      fp8_release_text(output, scale.tensor().as_int().to_string(), maximum)
    }
    ActivationInput(reference) => {
      fp8_release_text(output, "activation-input", maximum)
      fp8_release_text(output, reference.as_int().to_string(), maximum)
    }
    ActivationOutput(reference) => {
      fp8_release_text(output, "activation-output", maximum)
      fp8_release_text(output, reference.as_int().to_string(), maximum)
    }
    RuntimeInput(input) =>
      fp8_release_text(
        output,
        match input {
          StepCounts => "runtime:step-counts"
          QueryPositions => "runtime:query-positions"
          QueryRowOffsets => "runtime:query-row-offsets"
          SequenceLengths => "runtime:sequence-lengths"
          PageTableOffsets => "runtime:page-table-offsets"
          PageTablePageIndices => "runtime:page-table-page-indices"
        },
        maximum,
      )
    PersistentKvComponent(layer, component) => {
      fp8_release_text(output, "persistent-kv", maximum)
      fp8_release_text(output, layer.as_int().to_string(), maximum)
      fp8_release_text(
        output,
        if component is Key {
          "key"
        } else {
          "value"
        },
        maximum,
      )
    }
    Workspace => fp8_release_text(output, "workspace", maximum)
  }
}

///|
fn encode_fp8_release_manifest_v2(
  identity : @spec.ModelIdentity,
  target : @catalog.DeviceTarget,
  runtime_recipe_digest_sha256 : String,
  operations : ReadOnlyArray[LunaFp8ReleaseOperationV2],
  maximum : Int,
) -> Bytes raise LunaCapabilityManifestError {
  let output : Array[Byte] = []
  fp8_release_text(output, "lunaflux.luna-fp8-release-manifest.v2", maximum)
  fp8_release_text(output, identity.content().as_hex(), maximum)
  fp8_release_text(output, identity.plan().as_hex(), maximum)
  fp8_release_text(output, target.compute_major().to_string(), maximum)
  fp8_release_text(output, target.compute_minor().to_string(), maximum)
  fp8_release_text(output, target.supports_bf16().to_string(), maximum)
  fp8_release_text(output, target.supports_cublas_lt().to_string(), maximum)
  fp8_release_text(output, runtime_recipe_digest_sha256, maximum)
  fp8_release_text(output, operations.length().to_string(), maximum)
  for ordinal, operation in operations {
    fp8_release_text(output, ordinal.to_string(), maximum)
    fp8_release_text(output, operation.operation.as_int().to_string(), maximum)
    fp8_release_text(
      output,
      operation.entry_point.family().family_id().as_int().to_string(),
      maximum,
    )
    fp8_release_text(
      output,
      operation.entry_point.family().module_digest().sha256(),
      maximum,
    )
    fp8_release_text(
      output,
      operation.entry_point.entry_point_id().as_int().to_string(),
      maximum,
    )
    fp8_release_text(output, operation.module_locator, maximum)
    fp8_release_text(output, operation.function_symbol, maximum)
    fp8_release_text(output, operation.source_digest_sha256, maximum)
    fp8_release_text(output, operation.recipe_digest_sha256, maximum)
    let dimensions = operation.dimensions
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
    fp8_release_text(output, operation.operands.length().to_string(), maximum)
    for operand_ordinal, operand in operation.operands {
      fp8_release_text(output, operand_ordinal.to_string(), maximum)
      fp8_release_role(output, operand.role(), maximum)
      fp8_release_text(output, operand.byte_count().to_string(), maximum)
      fp8_release_text(output, operand.alignment().to_string(), maximum)
    }
    fp8_release_text(output, operation.workspace_bytes.to_string(), maximum)
    fp8_release_text(output, operation.workspace_alignment.to_string(), maximum)
  }
  Bytes::from_array(output)
}

///|
pub fn LunaFp8ReleaseManifestV2::new(
  identity~ : @spec.ModelIdentity,
  target~ : @catalog.DeviceTarget,
  runtime_recipe_digest_sha256~ : String,
  operations~ : ArrayView[LunaFp8ReleaseOperationV2],
  maximum_canonical_bytes~ : Int,
) -> LunaFp8ReleaseManifestV2 raise LunaCapabilityManifestError {
  if !luna_capability_is_sha256(runtime_recipe_digest_sha256) ||
    operations.is_empty() ||
    operations.length() > 65536 ||
    maximum_canonical_bytes <= 0 ||
    maximum_canonical_bytes > 16 * 1024 * 1024 {
    raise Invalid(Evidence)
  }
  let retained = ReadOnlyArray::from_array(operations)
  let canonical = encode_fp8_release_manifest_v2(
    identity, target, runtime_recipe_digest_sha256, retained, maximum_canonical_bytes,
  )
  {
    identity,
    target,
    runtime_recipe_digest_sha256,
    operations: retained,
    canonical,
    digest: LunaCapabilityManifestDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ),
  }
}
