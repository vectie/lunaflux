///|
fn approved_header_limits() -> @safetensors.HeaderLimits raise {
  @safetensors.HeaderLimits::new(
    max_header_bytes=4096,
    max_tensor_count=21,
    max_rank=2,
    max_name_length=128,
    max_payload_bytes=208544L,
    max_materialized_tensor_bytes=208544L,
  )
}

///|
fn load_release_bind_inputs(
  model_root_path : String,
) -> ReleaseBindInputs raise LunaBf16ReleaseBindError {
  let root = @approved_fs.ApprovedRoot::open_absolute(model_root_path) catch {
    error => {
      ignore(error)
      raise ModelAdmission
    }
  }
  let result : Result[ReleaseBindInputs, LunaBf16ReleaseBindError] = Ok(
    {
      let metadata = @config_file.load(
        root,
        @approved_fs.ApprovedRelativeLocator::new("config.json"),
        @config_file.ModelConfigDigest::from_sha256(APPROVED_CONFIG_SHA256),
        @spec.ContentDigest::from_sha256(APPROVED_MODEL_SHA256),
        @config_file.ModelConfigFileLimits::new(
          maximum_bytes=65536L,
          maximum_depth=16,
        ),
      )
      let plan = @llama.build_paged(
        metadata,
        batch=@llama.LlamaPagedBatchEnvelope::single_row(),
      )
      let weights = @device_materialize.inspect_file(
        root,
        metadata,
        plan,
        @approved_fs.ApprovedRelativeLocator::new("model.safetensors"),
        limits=@device_materialize.DeviceWeightFileLimits::new(
          max_file_bytes=210712L,
          transfer_chunk_bytes=4096,
          safetensors=approved_header_limits(),
        ),
        alignment_bytes=256L,
        max_arena_bytes=1048576L,
      )
      if weights.content_digest().as_hex() != APPROVED_MODEL_SHA256 {
        raise ModelAdmission
      }
      { plan, weights }
    },
  ) catch {
    error => {
      ignore(error)
      Err(ModelAdmission)
    }
  }
  let close_result : Result[Unit, LunaBf16ReleaseBindError] = Ok(root.close()) catch {
    error => {
      ignore(error)
      Err(ModelAdmission)
    }
  }
  match (result, close_result) {
    (Ok(value), Ok(_)) => value
    _ => raise ModelAdmission
  }
}
