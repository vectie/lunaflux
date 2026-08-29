///|
fn load_release_bind_inputs(
  arguments : Qwen3ReleaseBindArguments,
) -> Qwen3ReleaseBindInputs raise Qwen3Bf16ReleaseBindError {
  let root = @approved_fs.ApprovedRoot::open_absolute(arguments.model_root) catch {
    error => {
      ignore(error)
      raise ModelAdmission
    }
  }
  let result : Result[Qwen3ReleaseBindInputs, Qwen3Bf16ReleaseBindError] = Ok(
    {
      let file = root.open_file(
        @approved_fs.ApprovedRelativeLocator::new(arguments.config_locator),
      )
      let config = file.read_immutable_snapshot(maximum_bytes=1048576L)
      file.close()
      if @crypto.bytes_to_hex_string(@crypto.sha256(config)) !=
        arguments.config_sha256 {
        raise ModelAdmission
      }
      let metadata = @config_reader.parse_qwen3_bytes(
        config,
        @spec.ContentDigest::from_sha256(arguments.model_content_sha256),
        max_bytes=1048576,
        max_depth=16,
      )
      let batch = @qwen3.Qwen3PagedBatchEnvelope::single_row()
      let plan = @qwen3.build_paged(metadata, batch~)
      let weights = @qwen3_weights.inspect_paged_file(
        root,
        metadata,
        @approved_fs.ApprovedRelativeLocator::new(
          arguments.numeric_weights_locator,
        ),
        expected_artifact_digest=@numeric_weight_file.NumericWeightArtifactDigest::from_sha256(
          arguments.numeric_weight_artifact_sha256,
        ),
        limits=@numeric_weight_file.NumericWeightFileLimits::new(
          max_file_bytes=3221225472L,
          max_header_bytes=16777216,
          transfer_chunk_bytes=4194304,
          max_tensor_count=310,
        ),
        alignment_bytes=256L,
        max_arena_bytes=3221225472L,
        expected_route_manifest_sha256=arguments.weight_route_manifest_sha256,
        batch~,
      )
      if plan.operations().length() != 283 ||
        plan.tensor_count() != 310 ||
        weights.identity() != plan.identity() {
        raise ModelAdmission
      }
      { plan, weights: weights.generic_inspection() }
    },
  ) catch {
    error => {
      ignore(error)
      Err(ModelAdmission)
    }
  }
  let close_result : Result[Unit, Qwen3Bf16ReleaseBindError] = Ok(root.close()) catch {
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
