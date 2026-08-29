///|
fn open_numeric_weight_file(
  root : @approved_fs.ApprovedRoot,
  locator : @approved_fs.ApprovedRelativeLocator,
) -> @approved_fs.ApprovedFile raise NumericWeightFileError {
  root.open_file(locator) catch {
    UnsupportedType(File) => raise UnsupportedFileType
    _ => raise FileUnavailable
  }
}

///|
fn read_numeric_weight_range(
  file : @approved_fs.ApprovedFile,
  destination : FixedArray[Byte],
  destination_offset : Int,
  position : Int64,
  byte_count : Int,
) -> Unit raise NumericWeightFileError {
  file.read_exactly_at(destination, destination_offset~, byte_count~, position~) catch {
    Truncated => raise FileTruncated
    _ => raise FileUnavailable
  }
}

///|
fn numeric_weight_stamp(
  file : @approved_fs.ApprovedFile,
) -> @approved_fs.ApprovedFileStamp raise NumericWeightFileError {
  file.stamp() catch {
    _ => raise FileUnavailable
  }
}

///|
fn close_numeric_weight_file(
  file : @approved_fs.ApprovedFile,
) -> Unit raise NumericWeightFileError {
  file.close() catch {
    _ => raise FileCloseFailed
  }
}

///|
fn[T] finish_numeric_file_body(
  outcome : Result[T, NumericWeightFileError],
  close : () -> Unit raise NumericWeightFileError,
) -> T raise NumericWeightFileError {
  match outcome {
    Err(primary) => {
      close() catch {
        _ => ()
      }
      raise primary
    }
    Ok(value) => {
      close()
      value
    }
  }
}

///|
fn[S] inspection_from_prepared(
  model_plan : @plan.ModelPlan,
  locator : @approved_fs.ApprovedRelativeLocator,
  expected_artifact_digest : NumericWeightArtifactDigest,
  limits : NumericWeightFileLimits,
  alignment_bytes : Int64,
  max_arena_bytes : Int64,
  prepared : PreparedNumericWeightFile[S],
) -> NumericWeightFileInspection {
  {
    model_plan,
    locator,
    expected_artifact_digest,
    limits,
    alignment_bytes,
    max_arena_bytes,
    payload_offset_bytes: prepared.payload_offset_bytes,
    numeric_schema_digest: model_plan.numeric_binding().digest(),
    layout: prepared.layout,
    regions: prepared.regions,
  }
}

///|
fn prepare_numeric_file_handle(
  file : @approved_fs.ApprovedFile,
  model_plan : @plan.ModelPlan,
  expected_artifact_digest : NumericWeightArtifactDigest,
  limits : NumericWeightFileLimits,
  alignment_bytes : Int64,
  max_arena_bytes : Int64,
) -> PreparedNumericWeightFile[@approved_fs.ApprovedFileStamp] raise NumericWeightFileError {
  let stamp = numeric_weight_stamp(file)
  prepare_numeric_weight_file(
    model_plan,
    expected_artifact_digest,
    stamp,
    stamp.byte_count(),
    limits,
    alignment_bytes,
    max_arena_bytes,
    (destination, destination_offset, position, byte_count) => {
      read_numeric_weight_range(
        file, destination, destination_offset, position, byte_count,
      )
    },
    () => numeric_weight_stamp(file),
  )
}

///|
priv enum ReadyNumericSourceClose[T] {
  NumericSourceReady(T)
  NumericSourceCleanupRequired(T, @device.DeviceError)
}

///|
fn[T] finish_ready_numeric_source_close(
  resource : T,
  close_source : () -> Unit raise NumericWeightFileError,
  close_resource : (T) -> Unit raise @device.DeviceError,
) -> ReadyNumericSourceClose[T] raise NumericWeightFileError {
  try close_source() catch {
    _ =>
      try close_resource(resource) catch {
        cleanup => NumericSourceCleanupRequired(resource, cleanup)
      } noraise {
        _ => raise FileCloseFailed
      }
  } noraise {
    _ => NumericSourceReady(resource)
  }
}

///|
/// Completely authenticates one strict numeric-safetensors descendant without
/// opening a device resource.
///
/// The full-file digest is independently pinned because the file metadata
/// embeds model content, plan, and schema identities and therefore cannot also
/// be its own content digest. Tensor names are exactly
/// `lunaflux.tensor.v1.<ordinal>` and payload bytes are contiguous in ordinal
/// order. Only exact BF16, F8_E4M3, I8, and F32 spellings admitted by the
/// schema exist.
pub fn inspect_file(
  root : @approved_fs.ApprovedRoot,
  model_plan : @plan.ModelPlan,
  locator : @approved_fs.ApprovedRelativeLocator,
  expected_artifact_digest~ : NumericWeightArtifactDigest,
  limits~ : NumericWeightFileLimits,
  alignment_bytes~ : Int64,
  max_arena_bytes~ : Int64,
) -> NumericWeightFileInspection raise NumericWeightFileError {
  let file = open_numeric_weight_file(root, locator)
  let outcome : Result[NumericWeightFileInspection, NumericWeightFileError] = Ok(
    {
      let prepared = prepare_numeric_file_handle(
        file, model_plan, expected_artifact_digest, limits, alignment_bytes, max_arena_bytes,
      )
      inspection_from_prepared(
        model_plan, locator, expected_artifact_digest, limits, alignment_bytes, max_arena_bytes,
        prepared,
      )
    },
  ) catch {
    error => Err(error)
  }
  finish_numeric_file_body(outcome, () => close_numeric_weight_file(file))
}

///|
/// Reopens and completely reauthenticates the private inspected descendant.
/// The result is still inert and owns no device allocation.
pub fn NumericWeightFileInspection::reauthenticate(
  self : NumericWeightFileInspection,
  root : @approved_fs.ApprovedRoot,
) -> NumericWeightFileInspection raise NumericWeightFileError {
  inspect_file(
    root,
    self.model_plan,
    self.locator,
    expected_artifact_digest=self.expected_artifact_digest,
    limits=self.limits,
    alignment_bytes=self.alignment_bytes,
    max_arena_bytes=self.max_arena_bytes,
  )
}

///|
/// Reauthenticates an inspection before allocation, then hashes the exact
/// bytes again while streaming every tensor directly into its final aligned
/// region. No conversion or runtime quantization is performed.
pub fn load_inspected_file(
  context : @device.Context,
  root : @approved_fs.ApprovedRoot,
  inspection : NumericWeightFileInspection,
) -> NumericDeviceWeightPreparation raise NumericWeightFileError {
  let file = open_numeric_weight_file(root, inspection.locator)
  let outcome : Result[
    GenericNumericPreparation[@device.Allocation],
    NumericWeightFileError,
  ] = Ok(
    {
      let prepared = prepare_numeric_file_handle(
        file,
        inspection.model_plan,
        inspection.expected_artifact_digest,
        inspection.limits,
        inspection.alignment_bytes,
        inspection.max_arena_bytes,
      )
      if !prepared_matches_inspection(prepared, inspection) {
        raise FileChanged
      }
      transfer_prepared_numeric_file(
        prepared,
        (destination, destination_offset, position, byte_count) => {
          read_numeric_weight_range(
            file, destination, destination_offset, position, byte_count,
          )
        },
        () => numeric_weight_stamp(file),
        arena_bytes => context.allocate(arena_bytes),
        (allocation, source, source_offset, destination_offset, byte_count) => {
          context.copy_from_fixed_host(
            allocation,
            source,
            source_offset.to_int64(),
            destination_offset,
            byte_count,
          )
        },
        allocation => allocation.close(),
      )
    },
  ) catch {
    error => Err(error)
  }
  match outcome {
    Err(primary) => {
      close_numeric_weight_file(file) catch {
        _ => ()
      }
      raise primary
    }
    Ok(GenericNumericCleanupRequired(resource, cause, stage, cleanup)) => {
      close_numeric_weight_file(file) catch {
        _ => ()
      }
      NumericWeightsCleanupRequired({
        allocation: resource,
        cause,
        stage,
        cleanup_failure: cleanup,
        open: true,
      })
    }
    Ok(GenericNumericReady(resource)) =>
      // ApprovedFile is definitively closed even when POSIX close reports a
      // failure. Only allocation cleanup can remain retryable.
      match
        finish_ready_numeric_source_close(
          resource,
          () => close_numeric_weight_file(file),
          allocation => allocation.close(),
        ) {
        NumericSourceReady(resource) =>
          NumericWeightsReady({
            allocation: resource,
            artifact_digest: inspection.expected_artifact_digest,
            numeric_schema_digest: inspection.numeric_schema_digest,
            layout: inspection.layout,
            open: true,
          })
        NumericSourceCleanupRequired(resource, cleanup) =>
          NumericWeightsCleanupRequired({
            allocation: resource,
            cause: FileCloseFailed,
            stage: SourceClose,
            cleanup_failure: cleanup,
            open: true,
          })
      }
  }
}
