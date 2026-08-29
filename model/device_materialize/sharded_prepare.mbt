///|
fn sharded_checked_add(
  left : Int64,
  right : Int64,
  reference : Int,
  segment : Int,
) -> Int64 raise DeviceWeightError {
  if left < 0L || right < 0L || left > 9223372036854775807L - right {
    raise InvalidShardedTransfer(reference~, segment~)
  }
  left + right
}

///|
fn build_authenticated_sharded_rank_plan(
  model : @spec.LlamaModelMetadata,
  semantic_plan : @plan.ModelPlan,
  bindings : @llama_weights.LlamaWeightBindings,
  rank : Int,
  world_size : Int,
  alignment_bytes : Int64,
  max_arena_bytes : Int64,
) -> @tensor_parallel_plan.TensorParallelRankPlan raise DeviceWeightError {
  if semantic_plan.identity().content() != model.content_digest() {
    raise TensorParallelPlanMismatch
  }
  @llama_tensor_parallel.build_rank_plan(
    semantic_plan,
    bindings,
    rank~,
    world_size~,
    alignment_bytes~,
    max_arena_bytes~,
  ) catch {
    error => raise InvalidTensorParallelPlan(error)
  }
}

///|
fn absolute_sharded_transfer(
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
  payload_offset_bytes : Int64,
  file_bytes : Int64,
  reference : Int,
  segment_index : Int64,
) -> AbsoluteShardedTransfer raise DeviceWeightError {
  let tensor_reference = @plan.TensorRef::new(reference) catch {
    _ =>
      raise InvalidShardedTransfer(reference~, segment=segment_index.to_int())
  }
  guard rank_plan.tensor(tensor_reference) is Some(placement) else {
    raise InvalidShardedTransfer(reference~, segment=segment_index.to_int())
  }
  let recipe = placement.transfer_recipe()
  let segment = recipe.segment(segment_index) catch {
    _ =>
      raise InvalidShardedTransfer(reference~, segment=segment_index.to_int())
  }
  let source = sharded_checked_add(
    payload_offset_bytes,
    segment.source_payload_offset_bytes(),
    reference,
    segment_index.to_int(),
  )
  let source_end = sharded_checked_add(
    source,
    segment.byte_length(),
    reference,
    segment_index.to_int(),
  )
  let destination_end = sharded_checked_add(
    segment.destination_offset_bytes(),
    segment.byte_length(),
    reference,
    segment_index.to_int(),
  )
  let region = placement.region()
  if source < payload_offset_bytes ||
    source_end > file_bytes ||
    segment.destination_offset_bytes() < region.offset_bytes() ||
    destination_end > region.end_offset_bytes() {
    raise InvalidShardedTransfer(reference~, segment=segment_index.to_int())
  }
  {
    source_offset_bytes: source,
    destination_offset_bytes: segment.destination_offset_bytes(),
    byte_length: segment.byte_length(),
  }
}

///|
fn validate_sharded_transfers(
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
  source_order : FixedArray[Int],
  payload_offset_bytes : Int64,
  file_bytes : Int64,
) -> Unit raise DeviceWeightError {
  if rank_plan.tensor_count() <= 0 ||
    rank_plan.arena_bytes() <= 0L ||
    rank_plan.materialized_bytes() <= 0L ||
    rank_plan.materialized_bytes() > rank_plan.arena_bytes() {
    raise TensorParallelPlanMismatch
  }
  let mut previous_source_end = payload_offset_bytes
  let mut materialized = 0L
  if source_order.length() != rank_plan.tensor_count() {
    raise TensorParallelPlanMismatch
  }
  for order_index in 0..<source_order.length() {
    let reference = source_order[order_index]
    let tensor_reference = try! @plan.TensorRef::new(reference)
    guard rank_plan.tensor(tensor_reference) is Some(placement) else {
      raise InvalidShardedTransfer(reference~, segment=0)
    }
    let recipe = placement.transfer_recipe()
    if recipe.segment_count() <= 0L {
      raise InvalidShardedTransfer(reference~, segment=0)
    }
    let mut tensor_bytes = 0L
    for segment_index in 0L..<recipe.segment_count() {
      let segment = absolute_sharded_transfer(
        rank_plan, payload_offset_bytes, file_bytes, reference, segment_index,
      )
      if segment.source_offset_bytes < previous_source_end ||
        segment.destination_offset_bytes !=
        placement.region().offset_bytes() + tensor_bytes {
        raise InvalidShardedTransfer(reference~, segment=segment_index.to_int())
      }
      previous_source_end = segment.source_offset_bytes + segment.byte_length
      tensor_bytes = sharded_checked_add(
        tensor_bytes,
        segment.byte_length,
        reference,
        segment_index.to_int(),
      )
    }
    if tensor_bytes != placement.extent().byte_length() ||
      tensor_bytes != placement.region().byte_length() {
      raise InvalidShardedTransfer(reference~, segment=0)
    }
    materialized = sharded_checked_add(materialized, tensor_bytes, reference, 0)
  }
  if materialized != rank_plan.materialized_bytes() {
    raise TensorParallelPlanMismatch
  }
}

///|
fn sharded_first_source_offset(
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
  reference : Int,
) -> Int64 raise DeviceWeightError {
  let tensor_reference = try! @plan.TensorRef::new(reference)
  guard rank_plan.tensor(tensor_reference) is Some(placement) else {
    raise InvalidShardedTransfer(reference~, segment=0)
  }
  let recipe = placement.transfer_recipe()
  recipe.source_payload_offset_bytes_at(0L) catch {
    _ => raise InvalidShardedTransfer(reference~, segment=0)
  }
}

///|
fn sharded_source_order(
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
) -> FixedArray[Int] raise DeviceWeightError {
  let order = FixedArray::make(rank_plan.tensor_count(), 0)
  for reference in 0..<rank_plan.tensor_count() {
    order[reference] = reference
  }
  for index in 1..<order.length() {
    let reference = order[index]
    let source = sharded_first_source_offset(rank_plan, reference)
    let mut position = index
    while position > 0 &&
          sharded_first_source_offset(rank_plan, order[position - 1]) > source {
      order[position] = order[position - 1]
      position = position - 1
    }
    order[position] = reference
  }
  order
}

///|
fn[S : Eq] prepare_sharded_streaming_load(
  model : @spec.LlamaModelMetadata,
  semantic_plan : @plan.ModelPlan,
  rank : Int,
  world_size : Int,
  alignment_bytes : Int64,
  max_arena_bytes : Int64,
  source_stamp : S,
  file_bytes : Int64,
  limits : DeviceWeightFileLimits,
  read : (FixedArray[Byte], Int, Int64, Int) -> Unit raise DeviceWeightError,
  observe_stamp : () -> S raise DeviceWeightError,
) -> PreparedShardedStreamingLoad[S] raise DeviceWeightError {
  if file_bytes <= SAFETENSORS_PREFIX_BYTES64 {
    raise FileTruncated
  }
  if file_bytes > limits.max_file_bytes() {
    raise FileTooLarge
  }
  let prefix_scratch = FixedArray::make(8, b'\x00')
  read_exact_chunk(read, prefix_scratch, 0L, 8)
  let prefix = Bytes::from_array(prefix_scratch[:])
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
  let scratch_bytes = if payload_offset_bytes >
    limits.transfer_chunk_bytes().to_int64() {
    payload_offset_bytes.to_int()
  } else {
    limits.transfer_chunk_bytes()
  }
  let scratch = FixedArray::make(scratch_bytes, b'\x00')
  read_exact_chunk(read, scratch, 0L, payload_offset_bytes.to_int())
  let prefix_and_header = Bytes::from_array(
    scratch[:payload_offset_bytes.to_int()],
  )
  let metadata = @safetensors_reader.read_header(
    prefix_and_header,
    payload_bytes=file_bytes - payload_offset_bytes,
    limits=limits.safetensors(),
  ) catch {
    error => raise InvalidSafetensors(error)
  }
  let bindings = @llama_weights.bind(model, semantic_plan, metadata) catch {
    error => raise InvalidLlamaWeights(error)
  }
  let rank_plan = build_authenticated_sharded_rank_plan(
    model, semantic_plan, bindings, rank, world_size, alignment_bytes, max_arena_bytes,
  )
  let source_order = sharded_source_order(rank_plan)
  validate_sharded_transfers(
    rank_plan, source_order, payload_offset_bytes, file_bytes,
  )
  let digest = hash_stream(
    file_bytes,
    limits.transfer_chunk_bytes(),
    scratch,
    read,
  )
  let verified_stamp = observe_stamp() catch {
    _ => raise FileChangedDuringLoad
  }
  if verified_stamp != source_stamp {
    raise FileChangedDuringLoad
  }
  if digest != model.content_digest().as_hex() {
    raise ContentDigestMismatch
  }
  { source_stamp, payload_offset_bytes, rank_plan, source_order, scratch }
}
