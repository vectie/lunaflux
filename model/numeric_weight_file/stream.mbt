///|
const NUMERIC_MAX_HEADER_WITH_PREFIX : Int = 2147483639

///|
fn preflight_numeric_weight_scalars(
  tensor_count : Int,
  limits : NumericWeightFileLimits,
) -> Unit raise NumericWeightFileError {
  if tensor_count < 0 || tensor_count > limits.max_tensor_count() {
    raise InvalidLimit(TensorCount)
  }
}

///|
fn hash_numeric_file(
  file_bytes : Int64,
  scratch : FixedArray[Byte],
  chunk_bytes : Int,
  read : (FixedArray[Byte], Int, Int64, Int) -> Unit raise NumericWeightFileError,
) -> String raise NumericWeightFileError {
  let digest = @crypto.SHA256::new()
  let mut position = 0L
  while position < file_bytes {
    let remaining = file_bytes - position
    let count = if remaining < chunk_bytes.to_int64() {
      remaining.to_int()
    } else {
      chunk_bytes
    }
    read_numeric_exact(read, scratch, position, count)
    digest.update_from_iter(scratch[:count].iter())
    position += count.to_int64()
  }
  @crypto.bytes_to_hex_string(digest.finalize())
}

///|
fn[S : Eq] prepare_numeric_weight_file(
  model_plan : @plan.ModelPlan,
  expected_artifact_digest : NumericWeightArtifactDigest,
  source_stamp : S,
  file_bytes : Int64,
  limits : NumericWeightFileLimits,
  alignment_bytes : Int64,
  max_arena_bytes : Int64,
  read : (FixedArray[Byte], Int, Int64, Int) -> Unit raise NumericWeightFileError,
  observe_stamp : () -> S raise NumericWeightFileError,
) -> PreparedNumericWeightFile[S] raise NumericWeightFileError {
  if file_bytes <= 8L {
    raise FileTruncated
  }
  if file_bytes > limits.max_file_bytes() {
    raise FileTooLarge
  }
  let prefix_scratch = FixedArray::make(8, b'\x00')
  read_numeric_exact(read, prefix_scratch, 0L, 8)
  let prefix = Bytes::from_array(prefix_scratch[:])
  let safetensors_limits = @safetensors.HeaderLimits::new(
    max_header_bytes=limits.max_header_bytes(),
    max_tensor_count=limits.max_tensor_count(),
    max_rank=16,
    max_name_length=128,
    max_payload_bytes=limits.max_file_bytes(),
    max_materialized_tensor_bytes=limits.max_file_bytes(),
  ) catch {
    _ => raise InvalidLimit(HeaderBytes)
  }
  let header_bytes = @safetensors_reader.declared_header_bytes(
    prefix,
    limits=safetensors_limits,
  ) catch {
    error => raise InvalidSafetensorsPrefix(error)
  }
  if header_bytes > NUMERIC_MAX_HEADER_WITH_PREFIX {
    raise InvalidLimit(HeaderBytes)
  }
  let payload_offset_bytes = 8L + header_bytes.to_int64()
  if payload_offset_bytes > file_bytes {
    raise FileTruncated
  }
  let tensor_count = model_plan.numeric_binding().schema().tensors().length()
  preflight_numeric_weight_scalars(tensor_count, limits)
  // Admit exact layout scalars and arithmetic before any header-sized scratch,
  // immutable snapshot, decoded JSON tree, or transfer-region allocation.
  let layout = @device_materialize.plan_numeric_layout(
    model_plan,
    alignment_bytes~,
    max_arena_bytes~,
  ) catch {
    error => raise InvalidLayout(error)
  }
  let scratch_bytes = if payload_offset_bytes >
    limits.transfer_chunk_bytes().to_int64() {
    payload_offset_bytes.to_int()
  } else {
    limits.transfer_chunk_bytes()
  }
  let scratch = FixedArray::make(scratch_bytes, b'\x00')
  read_numeric_exact(read, scratch, 0L, payload_offset_bytes.to_int())
  let header_snapshot = Bytes::from_array(
    scratch[:payload_offset_bytes.to_int()],
  )
  let regions = parse_numeric_header(
    header_snapshot,
    file_bytes - payload_offset_bytes,
    model_plan,
    layout,
    limits,
  )
  validate_numeric_payload(
    model_plan,
    regions,
    scratch,
    limits.transfer_chunk_bytes(),
    read,
  )
  let digest = hash_numeric_file(
    file_bytes,
    scratch,
    limits.transfer_chunk_bytes(),
    read,
  )
  let final_stamp = observe_stamp() catch { _ => raise FileChanged }
  if final_stamp != source_stamp {
    raise FileChanged
  }
  if digest != expected_artifact_digest.as_hex() {
    raise ArtifactDigestMismatch
  }
  {
    stamp: source_stamp,
    payload_offset_bytes,
    artifact_digest: expected_artifact_digest,
    layout,
    regions,
    limits,
    scratch,
  }
}

///|
fn[S] prepared_matches_inspection(
  prepared : PreparedNumericWeightFile[S],
  inspection : NumericWeightFileInspection,
) -> Bool {
  prepared.payload_offset_bytes == inspection.payload_offset_bytes &&
  prepared.artifact_digest == inspection.expected_artifact_digest &&
  prepared.layout == inspection.layout &&
  prepared.regions == inspection.regions
}

///|
fn[T] close_failed_numeric_resource(
  resource : T,
  close : (T) -> Unit raise @device.DeviceError,
) -> Result[T, @device.DeviceError] {
  try close(resource) catch {
    error => Err(error)
  } noraise {
    _ => Ok(resource)
  }
}

///|
priv enum GenericNumericPreparation[T] {
  GenericNumericReady(T)
  GenericNumericCleanupRequired(
    T,
    NumericWeightFileError,
    NumericWeightFailureStage,
    @device.DeviceError
  )
}

///|
fn[S : Eq, T] transfer_prepared_numeric_file(
  prepared : PreparedNumericWeightFile[S],
  read : (FixedArray[Byte], Int, Int64, Int) -> Unit raise NumericWeightFileError,
  observe_stamp : () -> S raise NumericWeightFileError,
  open : (Int64) -> T raise @device.DeviceError,
  copy : (T, FixedArray[Byte], Int, Int64, Int64) -> Unit raise @device.DeviceError,
  close : (T) -> Unit raise @device.DeviceError,
) -> GenericNumericPreparation[T] raise NumericWeightFileError {
  let resource = open(prepared.layout.arena_bytes()) catch {
    error => raise DeviceFailed(error)
  }
  let digest = @crypto.SHA256::new()
  let outcome : Result[Unit, NumericWeightFileError] = Ok(
    {
      let mut position = 0L
      while position < prepared.payload_offset_bytes {
        let remaining = prepared.payload_offset_bytes - position
        let count = if remaining <
          prepared.limits.transfer_chunk_bytes().to_int64() {
          remaining.to_int()
        } else {
          prepared.limits.transfer_chunk_bytes()
        }
        read_numeric_exact(read, prepared.scratch, position, count)
        digest.update_from_iter(prepared.scratch[:count].iter())
        position += count.to_int64()
      }
      for region in prepared.regions {
        let mut offset = 0L
        while offset < region.byte_length {
          let remaining = region.byte_length - offset
          let count = if remaining <
            prepared.limits.transfer_chunk_bytes().to_int64() {
            remaining.to_int()
          } else {
            prepared.limits.transfer_chunk_bytes()
          }
          read_numeric_exact(
            read,
            prepared.scratch,
            region.source_offset_bytes + offset,
            count,
          )
          digest.update_from_iter(prepared.scratch[:count].iter())
          copy(
            resource,
            prepared.scratch,
            0,
            region.destination_offset_bytes + offset,
            count.to_int64(),
          ) catch {
            error => raise DeviceFailed(error)
          }
          offset += count.to_int64()
        }
      }
      let final_stamp = observe_stamp() catch { _ => raise FileChanged }
      if final_stamp != prepared.stamp {
        raise FileChanged
      }
      if @crypto.bytes_to_hex_string(digest.finalize()) !=
        prepared.artifact_digest.as_hex() {
        raise ArtifactDigestMismatch
      }
    },
  ) catch {
    error => Err(error)
  }
  match outcome {
    Ok(_) => GenericNumericReady(resource)
    Err(cause) => {
      let stage = match cause {
        DeviceFailed(_) => Transfer
        FileChanged | ArtifactDigestMismatch => SourceRevalidation
        _ => Transfer
      }
      match close_failed_numeric_resource(resource, close) {
        Ok(_) => raise cause
        Err(cleanup) =>
          GenericNumericCleanupRequired(resource, cause, stage, cleanup)
      }
    }
  }
}
