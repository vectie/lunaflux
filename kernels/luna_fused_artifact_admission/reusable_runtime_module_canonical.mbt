///|
fn reusable_runtime_append_target(
  out : StringBuilder,
  target : @catalog.DeviceTarget,
) -> Unit {
  out.write_string("compute_major=" + target.compute_major().to_string() + "\n")
  out.write_string("compute_minor=" + target.compute_minor().to_string() + "\n")
  out.write_string(
    "supports_bf16=" + (if target.supports_bf16() { "1" } else { "0" }) + "\n",
  )
  out.write_string(
    "supports_cublas_lt=" +
    (if target.supports_cublas_lt() { "1" } else { "0" }) +
    "\n",
  )
}

///|
fn reusable_runtime_append_module(
  out : StringBuilder,
  ordinal : Int,
  module_ : ReusableFusedRuntimeModuleInput,
  metadata : Bool,
  ownership : Bool,
  decode_metadata : Bool,
  prefill_metadata : Bool,
) -> Unit raise FusedArtifactAdmissionError {
  let prefix = "module_" + ordinal.to_string() + "_"
  out.write_string(
    prefix + "abi=" + reusable_runtime_abi_name(module_.abi) + "\n",
  )
  out.write_string(
    prefix + "first_operation=" + module_.first_operation.to_string() + "\n",
  )
  out.write_string(
    prefix + "operation_count=" + module_.operation_count.to_string() + "\n",
  )
  out.write_string(prefix + "function_symbol=" + module_.symbol + "\n")
  let query_tile_rows = match module_.query_tile_rows {
    Some(rows) => rows
    None => 0
  }
  out.write_string(
    prefix + "query_tile_rows=" + query_tile_rows.to_string() + "\n",
  )
  if metadata {
    out.write_string(
      prefix +
      "query_metadata_version=" +
      module_.query_metadata_version.to_string() +
      "\n",
    )
  }
  if ownership {
    out.write_string(
      prefix + "heads_per_cta=" + module_.heads_per_cta.to_string() + "\n",
    )
  }
  if decode_metadata {
    let (law, rows, stages) = match module_.decode_execution {
      Some(descriptor) => {
        let (rows, stages) = descriptor.storage_geometry()
        (descriptor.numeric_law(), rows, stages)
      }
      None => ("-", 0, 0)
    }
    out.write_string(prefix + "decode_numeric_law=" + law + "\n")
    out.write_string(
      prefix + "decode_key_value_tile_rows=" + rows.to_string() + "\n",
    )
    out.write_string(
      prefix + "decode_pipeline_stages=" + stages.to_string() + "\n",
    )
  }
  if prefill_metadata {
    reusable_runtime_append_prefill_exponential(
      out,
      prefix,
      module_.prefill_exponential_law,
    )
  }
  out.write_string(
    prefix + "grid_x=" + module_.launch.grid_x().to_string() + "\n",
  )
  out.write_string(
    prefix + "grid_y=" + module_.launch.grid_y().to_string() + "\n",
  )
  out.write_string(
    prefix + "grid_z=" + module_.launch.grid_z().to_string() + "\n",
  )
  out.write_string(
    prefix + "block_x=" + module_.launch.block_x().to_string() + "\n",
  )
  out.write_string(
    prefix + "block_y=" + module_.launch.block_y().to_string() + "\n",
  )
  out.write_string(
    prefix + "block_z=" + module_.launch.block_z().to_string() + "\n",
  )
  out.write_string(
    prefix +
    "shared_bytes=" +
    module_.launch.shared_memory_bytes().to_string() +
    "\n",
  )
  out.write_string(
    prefix +
    "module_sha256=" +
    fused_artifact_digest(module_.module_bytes).as_hex() +
    "\n",
  )
  out.write_string(
    prefix +
    "module_hex=" +
    @crypto.bytes_to_hex_string(module_.module_bytes) +
    "\n",
  )
}
