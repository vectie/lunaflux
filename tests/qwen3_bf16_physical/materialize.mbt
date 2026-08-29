///|
fn qwen3_close_failed_numeric(
  failed : @numeric_weight_file.FailedNumericDeviceWeightPreparation,
) -> @numeric_weight_file.NumericDeviceWeights raise Qwen3PhysicalError {
  ignore(failed.stage())
  failed.retry_close() catch {
    _ =>
      raise NumericMaterialization("failed numeric allocation could not close")
  }
  raise NumericMaterialization("numeric materialization failed after cleanup")
}

///|
async fn qwen3_materialize(
  args : ArrayView[String],
) -> Unit raise Qwen3PhysicalError {
  qwen3_require(args.length() == 10, "materialize arguments are invalid")
  let model_root_path = @fs.realpath(args[2]) catch {
    _ => raise InvalidInput("model root is unavailable")
  }
  qwen3_require(model_root_path == args[2], "model root is not canonical")
  let metadata = qwen3_load_metadata(model_root_path, args[3], args[4])
  let root = @approved_fs.ApprovedRoot::open_absolute(model_root_path) catch {
    _ => raise NumericMaterialization("approved model root open failed")
  }
  let inspection = @qwen3_weights.inspect_paged_file(
    root,
    metadata,
    @approved_fs.ApprovedRelativeLocator::new(args[5]),
    expected_artifact_digest=@numeric_weight_file.NumericWeightArtifactDigest::from_sha256(
      args[6],
    ),
    limits=@numeric_weight_file.NumericWeightFileLimits::new(
      max_file_bytes=68719476736L,
      max_header_bytes=16777216,
      transfer_chunk_bytes=16777216,
      max_tensor_count=65536,
    ),
    alignment_bytes=256L,
    max_arena_bytes=68719476736L,
    expected_route_manifest_sha256=args[7],
  ) catch {
    _ => {
      root.close() catch {
        _ => ()
      }
      raise NumericMaterialization(
        "canonical numeric artifact inspection failed",
      )
    }
  }
  qwen3_require(
    inspection.route_manifest_sha256() == args[7],
    "route manifest binding differs",
  )
  let context = @device.open_context(0) catch {
    _ => {
      root.close() catch {
        _ => ()
      }
      raise NumericMaterialization("CUDA context open failed")
    }
  }
  let identity = context.device_identity()
  if identity.uuid() != args[8] || identity.pci() != args[9] {
    context.close() catch {
      _ => ()
    }
    root.close() catch {
      _ => ()
    }
    raise DeviceMismatch("CUDA device identity differs")
  }
  let preparation = @qwen3_weights.load_inspected_paged_file(
    context, root, inspection,
  ) catch {
    _ => {
      context.close() catch {
        _ => ()
      }
      root.close() catch {
        _ => ()
      }
      raise NumericMaterialization("numeric loader rejected the artifact")
    }
  }
  let weights = match preparation {
    NumericWeightsReady(owner) => owner
    NumericWeightsCleanupRequired(failed) => qwen3_close_failed_numeric(failed)
  }
  let arena_bytes = weights.layout().arena_bytes()
  weights.close() catch {
    _ => raise NumericMaterialization("numeric device weights close failed")
  }
  context.close() catch {
    _ => raise NumericMaterialization("CUDA context close failed")
  }
  root.close() catch {
    _ => raise NumericMaterialization("approved model root close failed")
  }
  qwen3_require(weights.is_closed(), "numeric device weights remained open")
  println(
    "outcome=qwen3-numeric-materialization-pass artifact_sha256=" +
    args[6] +
    " route_manifest_sha256=" +
    args[7] +
    " arena_bytes=" +
    arena_bytes.to_string() +
    " device_uuid=" +
    args[8] +
    " device_pci=" +
    args[9] +
    " resources=root0,context0,allocation0,pending0,cleanup0",
  )
}
