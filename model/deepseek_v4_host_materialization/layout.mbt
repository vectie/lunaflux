///|
fn aligned64(value : Int) -> Int64 {
  (value.to_int64() + 63L) & -64L
}

///|
fn append_arena_size(
  sizes : Array[Int],
  size : Int,
  limits : DeepSeekV4HostMaterializationLimits,
) -> Unit raise DeepSeekV4HostMaterializationError {
  if size <= 0 {
    return
  }
  if sizes.length() >= limits.maximum_arenas {
    raise ArenaCountExceeded
  }
  sizes.push(size)
}

///|
fn build_layout(
  sources : ArrayView[HostLayoutSourceTensor],
  limits : DeepSeekV4HostMaterializationLimits,
) -> (
  ReadOnlyArray[DeepSeekV4HostLayoutTensor],
  ReadOnlyArray[Int],
  Int64,
  Int64,
) raise DeepSeekV4HostMaterializationError {
  if sources.length() > limits.maximum_index_entries {
    raise InvalidIndexCount(
      expected=limits.maximum_index_entries,
      actual=sources.length(),
    )
  }
  let layouts : Array[DeepSeekV4HostLayoutTensor] = []
  let sizes : Array[Int] = []
  let mut arena_ordinal = 0
  let mut arena_used = 0
  let mut total_tensor_bytes = 0L
  for binding_index, source in sources {
    if source.byte_count <= 0L ||
      source.byte_count > limits.maximum_arena_bytes.to_int64() {
      raise TensorTooLarge(binding_index)
    }
    if source.byte_count > limits.maximum_total_host_bytes - total_tensor_bytes {
      raise HostByteBudgetExceeded
    }
    total_tensor_bytes += source.byte_count
    let bytes = source.byte_count.to_int()
    let mut offset64 = aligned64(arena_used)
    if offset64 > (limits.maximum_arena_bytes - bytes).to_int64() {
      append_arena_size(sizes, arena_used, limits)
      arena_ordinal += 1
      arena_used = 0
      offset64 = 0L
    }
    let offset = offset64.to_int()
    layouts.push({
      binding_index,
      shard_ordinal: source.shard_ordinal,
      tensor_ordinal: source.tensor_ordinal,
      arena_ordinal,
      arena_offset: offset,
      byte_count: bytes,
      quantized_parameter_index: None,
    })
    arena_used = offset + bytes
  }
  append_arena_size(sizes, arena_used, limits)
  let mut total_arena_bytes = 0L
  for size in sizes {
    let size64 = size.to_int64()
    if size64 > limits.maximum_total_host_bytes - total_arena_bytes {
      raise HostByteBudgetExceeded
    }
    total_arena_bytes += size64
  }
  (
    ReadOnlyArray::from_array(layouts),
    ReadOnlyArray::from_array(sizes),
    total_tensor_bytes,
    total_arena_bytes,
  )
}

///|
fn scale_parameter_name(name : String) -> String? {
  if name.length() < 5 || name[name.length() - 5:] != "scale" {
    return None
  }
  Some(name[:name.length() - 5].to_owned() + "weight")
}

///|
fn attach_quantized_parameter_pairs(
  binding : @deepseek_v4_weights.DeepSeekV4WeightBinding,
  layouts : ReadOnlyArray[DeepSeekV4HostLayoutTensor],
) -> ReadOnlyArray[DeepSeekV4HostLayoutTensor] raise DeepSeekV4HostMaterializationError {
  let by_name : Map[String, Int] = Map([], capacity=binding.tensor_count())
  for index in 0..<binding.tensor_count() {
    guard binding.tensor(index) is Some(bound) else {
      raise WeightBindingFailure
    }
    by_name.set(bound.tensor().name(), index)
  }
  let paired : Array[DeepSeekV4HostLayoutTensor] = []
  for layout in layouts {
    guard binding.tensor(layout.binding_index) is Some(bound) else {
      raise WeightBindingFailure
    }
    let parameter_index = if bound.encoding() is Ue8M0Scale {
      guard scale_parameter_name(bound.tensor().name()) is Some(name) &&
        by_name.get(name) is Some(index) &&
        binding.tensor(index) is Some(parameter) &&
        quantized_parameter_pair(bound.semantic(), parameter.semantic()) else {
        raise QuantizedPairUnavailable(layout.binding_index)
      }
      Some(index)
    } else {
      None
    }
    paired.push({
      binding_index: layout.binding_index,
      shard_ordinal: layout.shard_ordinal,
      tensor_ordinal: layout.tensor_ordinal,
      arena_ordinal: layout.arena_ordinal,
      arena_offset: layout.arena_offset,
      byte_count: layout.byte_count,
      quantized_parameter_index: parameter_index,
    })
  }
  ReadOnlyArray::from_array(paired)
}
