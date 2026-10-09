///|
pub fn DeepSeekV4TokenHashMaterializationLimits::new(
  maximum_tables~ : Int,
  maximum_total_i32_bytes~ : Int64,
) -> Self raise DeepSeekV4TokenHashMaterializationError {
  if maximum_tables <= 0 ||
    maximum_tables > 1024 ||
    maximum_total_i32_bytes <= 0L ||
    maximum_total_i32_bytes > 2147483647L {
    raise TokenHashInvalidLimit
  }
  { maximum_tables, maximum_total_i32_bytes, }
}

///|
fn release_narrowed_tables(
  tables : Array[@checked_integer_narrow.CheckedI32Table?],
) -> Unit {
  for table in tables {
    if table is Some(value) {
      ignore(value.release())
    }
  }
}

///|
fn exact_plan_identity(
  plan : @deepseek_v4.DeepSeekV4SemanticPlan,
) -> @spec.ModelIdentity raise DeepSeekV4TokenHashMaterializationError {
  let execution = plan.execution_plan()
  @spec.ModelIdentity::new(execution.content_digest(), execution.plan_digest()) catch {
    _ => raise TokenHashIdentityMismatch
  }
}

///|
/// Builds the official first-layer token-hash sidecars without retaining an
/// I64 copy. Each source tensor is read directly from its final authenticated
/// host arena, checked for expert range and per-row uniqueness, then narrowed
/// into one independently releasable I32 table.
pub fn DeepSeekV4HostWeights::materialize_token_hash_tables(
  self : Self,
  plan~ : @deepseek_v4.DeepSeekV4SemanticPlan,
  limits~ : DeepSeekV4TokenHashMaterializationLimits,
) -> DeepSeekV4TokenHashTables raise DeepSeekV4TokenHashMaterializationError {
  if self.released {
    raise TokenHashSourceReleased
  }
  let identity = exact_plan_identity(plan)
  if identity != self.identity {
    raise TokenHashIdentityMismatch
  }
  let execution = plan.execution_plan()
  let shape = execution.shape()
  let moe = execution.moe()
  let hash_layer_count = moe.hash_routed_layer_count()
  if hash_layer_count <= 0 || hash_layer_count > limits.maximum_tables {
    raise TokenHashTableCountMismatch(expected=hash_layer_count, actual=0)
  }
  let per_table_elements = shape.vocabulary_size().to_int64() *
    moe.experts_per_token().to_int64()
  if per_table_elements <= 0L ||
    per_table_elements > 2147483647L / 4L ||
    hash_layer_count.to_int64() >
    limits.maximum_total_i32_bytes / (per_table_elements * 4L) {
    raise TokenHashOutputBudgetExceeded
  }
  let narrowed : Array[@checked_integer_narrow.CheckedI32Table?] = Array::make(
    hash_layer_count,
    None,
  )
  let source_indices = Array::make(hash_layer_count, -1)
  let mut actual_count = 0
  for tensor_index in 0..<self.tensors.length() {
    let host_tensor = self.tensors[tensor_index]
    let binding = host_tensor.binding
    if binding.semantic().role() is TokenExpertHash {
      guard binding.semantic().scope() is DecoderLayer(layer_index) else {
        release_narrowed_tables(narrowed)
        raise TokenHashInvalidBinding(tensor_index)
      }
      if layer_index < 0 || layer_index >= hash_layer_count {
        release_narrowed_tables(narrowed)
        raise TokenHashInvalidBinding(tensor_index)
      }
      if narrowed[layer_index] is Some(_) {
        release_narrowed_tables(narrowed)
        raise TokenHashDuplicateLayer(layer_index)
      }
      let tensor = binding.tensor()
      if binding.encoding() != Int64 ||
        tensor.dtype() != I64 ||
        tensor.shape() !=
        [shape.vocabulary_size().to_int64(), moe.experts_per_token().to_int64()] ||
        host_tensor.byte_count.to_int64() != per_table_elements * 8L {
        release_narrowed_tables(narrowed)
        raise TokenHashInvalidBinding(tensor_index)
      }
      guard self.arenas[host_tensor.arena_ordinal] is Some(arena) else {
        release_narrowed_tables(narrowed)
        raise TokenHashSourceReleased
      }
      guard arena.get_view(
          start=host_tensor.arena_offset,
          end=host_tensor.arena_offset + host_tensor.byte_count,
        )
        is Some(source) else {
        release_narrowed_tables(narrowed)
        raise TokenHashInvalidBinding(tensor_index)
      }
      let table = @checked_integer_narrow.narrow_unique_i64_le_rows(
        source,
        row_count=shape.vocabulary_size(),
        column_count=moe.experts_per_token(),
        minimum_value=0,
        exclusive_maximum_value=moe.routed_expert_count(),
        maximum_output_bytes=(per_table_elements * 4L).to_int(),
      ) catch {
        error => {
          release_narrowed_tables(narrowed)
          raise TokenHashNarrowingFailure(layer=layer_index, error)
        }
      }
      narrowed[layer_index] = Some(table)
      source_indices[layer_index] = tensor_index
      actual_count += 1
    }
  }
  if actual_count != hash_layer_count {
    release_narrowed_tables(narrowed)
    raise TokenHashTableCountMismatch(
      expected=hash_layer_count,
      actual=actual_count,
    )
  }
  let entries : Array[DeepSeekV4TokenHashTableEntry] = []
  for layer_index in 0..<hash_layer_count {
    guard narrowed[layer_index] is Some(table) else {
      release_narrowed_tables(narrowed)
      raise TokenHashMissingLayer(layer_index)
    }
    entries.push({
      layer_index,
      source_tensor_index: source_indices[layer_index],
      table,
    })
  }
  {
    identity,
    source_manifest_digest: self.manifest_digest,
    entries: ReadOnlyArray::from_array(entries),
    released: false,
  }
}
