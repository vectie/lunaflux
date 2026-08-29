///|
fn append_field(output : Array[Byte], value : String) -> Unit {
  let encoded = @utf8.encode(value)
  for byte in encoded.length().to_int64().reinterpret_as_uint64().to_le_bytes() {
    output.push(byte)
  }
  for byte in encoded {
    output.push(byte)
  }
}

///|
fn append_shape(output : Array[Byte], shape : Fp8ExactExecutionShape) -> Unit {
  match shape.data {
    Stateless(profile, exact) => {
      append_field(output, "stateless")
      append_field(output, profile.as_int().to_string())
      append_field(output, exact.batch_rows().to_string())
      append_field(output, exact.query_tokens_per_row().to_string())
      append_field(output, exact.context_tokens().to_string())
      append_field(output, exact.token_rows().to_string())
    }
    Paged(profile, rows, query_tokens, page_entries) => {
      append_field(output, "paged")
      append_field(output, profile.as_int().to_string())
      append_field(output, rows.length().to_string())
      append_field(output, query_tokens.to_string())
      append_field(output, page_entries.to_string())
      for row in rows {
        append_field(output, row.query_tokens.to_string())
        append_field(output, row.context_tokens.to_string())
        append_field(output, row.page_table_entries.to_string())
      }
    }
  }
}

///|
fn append_region(output : Array[Byte], region : Fp8LaunchStorageRegion) -> Unit {
  append_field(output, region.storage.0)
  append_field(output, region.storage_bytes.to_string())
  append_field(output, region.offset.to_string())
  append_field(output, region.byte_length.to_string())
  append_field(output, region.alignment.to_string())
}

///|
fn encode_abi(
  plan : @plan.ModelPlan,
  manifest : @numeric_capability_manifest.Fp8KernelCapabilityManifest,
  entry : @numeric_capability_manifest.Fp8KernelManifestEntry,
  shape : Fp8ExactExecutionShape,
  scale_tensor_regions : ReadOnlyArray[Fp8ScaleTensorStorageRegion],
  activation_scale : Fp8LaunchStorageRegion,
  kernel_workspace : Fp8LaunchStorageRegion?,
  workspace_bytes : Int64,
  workspace_alignment : Int64,
) -> Fp8LaunchAbiDigest {
  let output : Array[Byte] = []
  append_field(output, "lunaflux.fp8.dynamic-launch-abi.v1")
  append_field(output, plan.identity().content().as_hex())
  append_field(output, plan.identity().plan().as_hex())
  append_field(output, plan.numeric_binding().digest().as_hex())
  append_field(output, manifest.digest().as_hex())
  append_field(output, manifest.target().compute_major().to_string())
  append_field(output, manifest.target().compute_minor().to_string())
  append_field(output, entry.profile().as_int().to_string())
  append_field(output, entry.operation().as_int().to_string())
  append_field(output, entry.execution_digest().as_hex())
  append_field(output, entry.module_digest().sha256())
  append_field(
    output,
    entry.entry_point().family().family_id().as_int().to_string(),
  )
  append_field(
    output,
    entry.entry_point().entry_point_id().as_int().to_string(),
  )
  append_shape(output, shape)
  append_field(output, "dynamic-per-tensor-f32-v1")
  append_field(output, entry.weights().length().to_string())
  for index, weight in entry.weights() {
    append_field(output, weight.tensor().as_int().to_string())
    append_field(output, weight.storage_digest().as_hex())
    append_field(output, weight.scale().tensor().as_int().to_string())
    append_field(output, weight.scale_storage_digest().as_hex())
    append_region(output, scale_tensor_regions[index].region)
  }
  append_region(output, activation_scale)
  append_field(output, workspace_bytes.to_string())
  append_field(output, workspace_alignment.to_string())
  match kernel_workspace {
    None => append_field(output, "workspace-absent")
    Some(region) => append_region(output, region)
  }
  Fp8LaunchAbiDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(Bytes::from_array(output))),
  )
}
