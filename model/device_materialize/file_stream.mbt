///|
const SAFETENSORS_PREFIX_BYTES64 : Int64 = 8L

///|
const MAX_HEADER_WITH_PREFIX_BYTES : Int = 2147483639

///|
async fn read_exact_chunk(
  read : async (Int64, Int) -> Bytes,
  position : Int64,
  byte_count : Int,
) -> Bytes raise DeviceWeightError {
  let bytes = read(position, byte_count) catch { _ => raise FileTruncated }
  if bytes.length() != byte_count {
    raise FileTruncated
  }
  bytes
}

///|
async fn hash_stream(
  file_bytes : Int64,
  chunk_bytes : Int,
  read : async (Int64, Int) -> Bytes,
) -> String raise DeviceWeightError {
  let digest = @crypto.SHA256::new()
  let mut position = 0L
  while position < file_bytes {
    let remaining = file_bytes - position
    let count = if remaining < chunk_bytes.to_int64() {
      remaining.to_int()
    } else {
      chunk_bytes
    }
    let bytes = read_exact_chunk(read, position, count)
    digest.update(bytes)
    position = position + count.to_int64()
  }
  @crypto.bytes_to_hex_string(digest.finalize())
}

///|
fn compare_transfer_source(
  left : StreamingTransferRegion,
  right : StreamingTransferRegion,
) -> Int {
  if left.source_offset_bytes < right.source_offset_bytes {
    -1
  } else if left.source_offset_bytes > right.source_offset_bytes {
    1
  } else {
    0
  }
}

///|
fn streaming_regions(
  bindings : @llama_weights.LlamaWeightBindings,
  layout : DeviceWeightLayout,
  payload_offset_bytes : Int64,
  file_bytes : Int64,
) -> ReadOnlyArray[StreamingTransferRegion] raise DeviceWeightError {
  let regions : Array[StreamingTransferRegion] = []
  for index in 0..<bindings.length() {
    let reference = try! @plan.TensorRef::new(index)
    guard bindings.binding(reference) is Some(binding) else {
      raise MissingBinding(reference=index)
    }
    guard layout.region(reference) is Some(destination) else {
      raise MissingBinding(reference=index)
    }
    let tensor = binding.tensor()
    if tensor.byte_length() != destination.byte_length() {
      raise TensorLengthMismatch(reference=index)
    }
    let source_offset = payload_offset_bytes + tensor.data_start()
    if source_offset < payload_offset_bytes ||
      tensor.byte_length() > file_bytes - source_offset {
      raise TensorLengthMismatch(reference=index)
    }
    regions.push({
      reference,
      source_offset_bytes: source_offset,
      destination_offset_bytes: destination.offset_bytes(),
      byte_length: tensor.byte_length(),
    })
  }
  regions.sort_by(compare_transfer_source)
  let mut expected_source = payload_offset_bytes
  for region in regions {
    if region.source_offset_bytes != expected_source {
      raise TensorLengthMismatch(reference=region.reference.as_int())
    }
    expected_source = expected_source + region.byte_length
  }
  if expected_source != file_bytes {
    raise TensorLengthMismatch(reference=0)
  }
  ReadOnlyArray::from_array(regions)
}

///|
async fn prepare_streaming_load(
  model : @spec.LlamaModelMetadata,
  source_stamp : SourceStamp,
  limits : DeviceWeightFileLimits,
  alignment_bytes : Int64,
  max_arena_bytes : Int64,
  read : async (Int64, Int) -> Bytes,
  observe_stamp : async () -> SourceStamp,
) -> PreparedStreamingLoad raise DeviceWeightError {
  let file_bytes = source_stamp.file_bytes
  if file_bytes <= SAFETENSORS_PREFIX_BYTES64 {
    raise FileTruncated
  }
  if file_bytes > limits.max_file_bytes() {
    raise FileTooLarge
  }
  let prefix = read_exact_chunk(read, 0L, 8)
  let header_bytes = @safetensors_reader.declared_header_bytes(
    prefix,
    limits=limits.safetensors(),
  ) catch {
    error => raise InvalidSafetensors(error)
  }
  if header_bytes > MAX_HEADER_WITH_PREFIX_BYTES {
    raise InvalidSafetensors(HeaderLengthOverflow)
  }
  let payload_offset_bytes = SAFETENSORS_PREFIX_BYTES64 +
    header_bytes.to_int64()
  if payload_offset_bytes > file_bytes {
    raise InvalidSafetensors(TruncatedHeader)
  }
  let prefix_and_header = read_exact_chunk(
    read,
    0L,
    payload_offset_bytes.to_int(),
  )
  let metadata = @safetensors_reader.read_header(
    prefix_and_header,
    payload_bytes=file_bytes - payload_offset_bytes,
    limits=limits.safetensors(),
  ) catch {
    error => raise InvalidSafetensors(error)
  }
  let bindings = @llama_weights.bind(model, metadata) catch {
    error => raise InvalidLlamaWeights(error)
  }
  let layout = plan_layout(bindings, alignment_bytes~, max_arena_bytes~)
  let regions = streaming_regions(
    bindings, layout, payload_offset_bytes, file_bytes,
  )
  let verified_digest = hash_stream(
    file_bytes,
    limits.transfer_chunk_bytes(),
    read,
  )
  let verified_stamp = observe_stamp() catch {
    _ => raise FileChangedDuringLoad
  }
  if verified_stamp != source_stamp {
    raise FileChangedDuringLoad
  }
  if verified_digest != model.identity().content().as_hex() {
    raise ContentDigestMismatch
  }
  { source_stamp, payload_offset_bytes, layout, regions }
}

///|
async fn[T] transfer_streaming_load(
  prepared : PreparedStreamingLoad,
  expected_digest : String,
  chunk_bytes : Int,
  resource : T,
  read : async (Int64, Int) -> Bytes,
  observe_stamp : async () -> SourceStamp,
  copy : (T, Bytes, Int64, Int64) -> Unit raise @device.DeviceError,
) -> Unit raise DeviceWeightError {
  let digest = @crypto.SHA256::new()
  let mut header_position = 0L
  while header_position < prepared.payload_offset_bytes {
    let remaining = prepared.payload_offset_bytes - header_position
    let count = if remaining < chunk_bytes.to_int64() {
      remaining.to_int()
    } else {
      chunk_bytes
    }
    let bytes = read_exact_chunk(read, header_position, count)
    digest.update(bytes)
    header_position = header_position + count.to_int64()
  }
  for region in prepared.regions {
    let mut transferred = 0L
    while transferred < region.byte_length {
      let remaining = region.byte_length - transferred
      let count = if remaining < chunk_bytes.to_int64() {
        remaining.to_int()
      } else {
        chunk_bytes
      }
      let bytes = read_exact_chunk(
        read,
        region.source_offset_bytes + transferred,
        count,
      )
      digest.update(bytes)
      copy(
        resource,
        bytes,
        region.destination_offset_bytes + transferred,
        count.to_int64(),
      ) catch {
        error => raise DeviceFailed(error)
      }
      transferred = transferred + count.to_int64()
    }
  }
  let observed_stamp = observe_stamp() catch {
    _ => raise FileChangedDuringLoad
  }
  if observed_stamp != prepared.source_stamp ||
    @crypto.bytes_to_hex_string(digest.finalize()) != expected_digest {
    raise FileChangedDuringLoad
  }
}

///|
async fn[T] load_streaming_with_resource(
  model : @spec.LlamaModelMetadata,
  source_stamp : SourceStamp,
  limits : DeviceWeightFileLimits,
  alignment_bytes : Int64,
  max_arena_bytes : Int64,
  read : async (Int64, Int) -> Bytes,
  observe_stamp : async () -> SourceStamp,
  open : (Int64) -> T raise @device.DeviceError,
  copy : (T, Bytes, Int64, Int64) -> Unit raise @device.DeviceError,
  close : (T) -> Unit raise @device.DeviceError,
) -> LoadedResource[T] raise DeviceWeightError {
  let prepared = prepare_streaming_load(
    model, source_stamp, limits, alignment_bytes, max_arena_bytes, read, observe_stamp,
  )
  let resource = open(prepared.layout.arena_bytes()) catch {
    error => raise DeviceFailed(error)
  }
  let transfer_result : Result[Unit, DeviceWeightError] = try {
    transfer_streaming_load(
      prepared,
      model.identity().content().as_hex(),
      limits.transfer_chunk_bytes(),
      resource,
      read,
      observe_stamp,
      copy,
    )
    Ok(())
  } catch {
    error => Err(error)
  }
  match transfer_result {
    Ok(_) => { resource, layout: prepared.layout }
    Err(error) => {
      let stage = match error {
        DeviceFailed(_) => Transfer
        _ => SourceRevalidation
      }
      close_after_failure(resource, stage, close)
      raise error
    }
  }
}
