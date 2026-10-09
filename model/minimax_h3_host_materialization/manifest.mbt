///|
fn index_claim(
  claims : ArrayView[MiniMaxH3TensorIndexInput],
  name : String,
) -> (Int, Int) {
  let mut count = 0
  let mut shard_index = -1
  for claim in claims {
    if claim.tensor_name == name {
      count += 1
      shard_index = claim.shard_index
    }
  }
  (count, shard_index)
}

///|
fn align64(value : Int64) -> Int64 {
  (value + 63L) & -64L
}

///|
fn append_arena(
  sizes : Array[Int64],
  size : Int64,
  limits : MiniMaxH3HostMaterializationLimits,
) -> Unit raise MiniMaxH3HostMaterializationError {
  if size <= 0L {
    return
  }
  if sizes.length() >= limits.maximum_arenas {
    raise ArenaCountExceeded
  }
  sizes.push(size)
}

///|
fn total_sizes(
  sizes : ArrayView[Int64],
  maximum : Int64,
) -> Int64 raise MiniMaxH3HostMaterializationError {
  let mut total = 0L
  for size in sizes {
    if size > maximum - total {
      raise HostBudgetExceeded
    }
    total += size
  }
  total
}

///|
/// Rejects components whose complete official tensor vocabulary is not yet
/// available. This check intentionally cannot admit a complete H3 model.
pub fn require_materializable_component(
  component : @minimax_h3_weights.MiniMaxH3WeightComponent,
) -> Unit {
  match component {
    Fl2vaDenoiser | Ref2vaDenoiser | TextConditioner | VideoVae | AudioVae => ()
  }
}

///|
/// Builds the exact component manifest and final bounded-arena layout.
pub fn build_manifest(
  binding : @minimax_h3_weights.MiniMaxH3CompleteComponentBinding,
  expected_identity~ : @spec.ModelIdentity,
  shards~ : ArrayView[MiniMaxH3ShardDeclarationInput],
  index~ : ArrayView[MiniMaxH3TensorIndexInput],
  limits~ : MiniMaxH3HostMaterializationLimits,
) -> MiniMaxH3HostComponentManifest raise MiniMaxH3HostMaterializationError {
  let component = binding.component()
  require_materializable_component(component)
  if binding.identity() != expected_identity {
    raise IdentityMismatch
  }
  let required = component.required_tensor_count()
  if binding.tensor_count() != required || required > limits.maximum_tensors {
    raise InvalidIndexCount(expected=required, actual=binding.tensor_count())
  }
  if shards.is_empty() || shards.length() > limits.maximum_shards {
    raise InvalidShardCount
  }
  for shard_index, shard in shards {
    if shard.payload_bytes > limits.maximum_payload_bytes_per_shard {
      raise ShardPayloadMismatch(shard_index)
    }
    for earlier in 0..<shard_index {
      if shards[earlier].file_digest == shard.file_digest {
        raise DuplicateShard(shard_index)
      }
    }
  }
  if index.length() != required {
    raise InvalidIndexCount(expected=required, actual=index.length())
  }
  let entries : Array[MiniMaxH3HostManifestTensor] = []
  let mut total_tensor_bytes = 0L
  let mut arena_index = 0
  let mut arena_used = 0L
  let arena_sizes : Array[Int64] = []
  for binding_index in 0..<binding.tensor_count() {
    guard binding.tensor(binding_index) is Some(bound) else {
      raise MissingIndex(binding_index)
    }
    let tensor = bound.tensor()
    if !exact_official_shape(component, tensor.name(), tensor.shape()) {
      raise InvalidOfficialShape(binding_index)
    }
    let (claim_count, shard_index) = index_claim(index, tensor.name())
    if claim_count == 0 {
      raise MissingIndex(binding_index)
    }
    if claim_count > 1 {
      raise DuplicateIndex(binding_index)
    }
    if shard_index < 0 || shard_index >= shards.length() {
      raise ShardIndexOutsideManifest(binding_index)
    }
    if tensor.data_start() < 0L ||
      tensor.data_end() < tensor.data_start() ||
      tensor.data_end() > shards[shard_index].payload_bytes {
      raise InvalidShardRange(binding_index)
    }
    for previous in entries {
      if previous.shard_index == shard_index &&
        previous.tensor.data_start() < tensor.data_end() &&
        tensor.data_start() < previous.tensor.data_end() {
        raise InvalidShardRange(binding_index)
      }
    }
    let byte_length = tensor.byte_length()
    if byte_length > limits.maximum_arena_bytes {
      raise TensorTooLarge(binding_index)
    }
    if byte_length > limits.maximum_total_host_bytes - total_tensor_bytes {
      raise HostBudgetExceeded
    }
    total_tensor_bytes += byte_length
    let mut offset = align64(arena_used)
    if offset > limits.maximum_arena_bytes - byte_length {
      append_arena(arena_sizes, arena_used, limits)
      arena_index += 1
      arena_used = 0L
      offset = 0L
    }
    entries.push({
      binding_index,
      role: bound.role(),
      tensor,
      shard_index,
      arena_index,
      arena_offset: offset,
    })
    arena_used = offset + byte_length
  }
  append_arena(arena_sizes, arena_used, limits)
  ignore(total_sizes(arena_sizes, limits.maximum_total_host_bytes))
  for claim_index, claim in index {
    if claim.tensor_name.length() > limits.maximum_name_bytes ||
      claim.shard_index < 0 ||
      claim.shard_index >= shards.length() {
      raise UnknownIndex(claim_index)
    }
    let mut found = false
    for entry in entries {
      if entry.tensor.name() == claim.tensor_name {
        found = true
      }
    }
    if !found {
      raise UnknownIndex(claim_index)
    }
  }
  for shard_index in 0..<shards.length() {
    let ordered : Array[MiniMaxH3HostManifestTensor] = []
    for entry in entries {
      if entry.shard_index == shard_index {
        ordered.push(entry)
      }
    }
    if ordered.is_empty() {
      raise UnreferencedShard(shard_index)
    }
    ordered.sort_by((left, right) => {
      if left.tensor.data_start() < right.tensor.data_start() {
        -1
      } else if left.tensor.data_start() > right.tensor.data_start() {
        1
      } else {
        0
      }
    })
    let mut expected_start = 0L
    for entry in ordered {
      if entry.tensor.data_start() != expected_start {
        raise InvalidShardRange(entry.binding_index)
      }
      expected_start = entry.tensor.data_end()
    }
    if expected_start != shards[shard_index].payload_bytes {
      raise ShardPayloadMismatch(shard_index)
    }
  }
  let owned_shards = ReadOnlyArray::from_array(shards)
  let owned_entries = ReadOnlyArray::from_array(entries)
  let owned_arena_sizes = ReadOnlyArray::from_array(arena_sizes)
  let digest = canonical_manifest_digest(
    component, expected_identity, owned_shards, owned_entries, owned_arena_sizes,
    total_tensor_bytes,
  )
  {
    component,
    identity: expected_identity,
    digest,
    shards: owned_shards,
    tensors: owned_entries,
    arena_sizes: owned_arena_sizes,
    total_tensor_bytes,
  }
}
