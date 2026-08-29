///|
fn llama_tp_append_collective(
  output : Array[@tensor_parallel_plan.TensorCollectiveSite],
  operation : Int,
  kind : @tensor_parallel_plan.TensorCollectiveKind,
  full_width : Int,
  local_width : Int,
  world_size : Int,
) -> Unit raise LlamaTensorParallelError {
  let operation_id = @plan.OperationId::new(operation) catch {
    _ => raise UnsupportedPlan(operation~)
  }
  output.push(
    @tensor_parallel_plan.TensorCollectiveSite::new(
      sequence=output.length() + 1,
      operation=operation_id,
      kind~,
      full_width~,
      local_width~,
      world_size~,
    ) catch {
      InvalidTensorParallelPlan(..) as error => raise InvalidPhysicalPlan(error)
    },
  )
}

///|
fn llama_tp_build_collectives(
  layer_count : Int,
  hidden : Int,
  vocabulary : Int,
  topology : @tensor_parallel_plan.TensorParallelTopology,
) -> ReadOnlyArray[@tensor_parallel_plan.TensorCollectiveSite] raise LlamaTensorParallelError {
  let output : Array[@tensor_parallel_plan.TensorCollectiveSite] = []
  if topology.world_size() == 1 {
    return ReadOnlyArray::from_array(output)
  }
  llama_tp_append_collective(
    output,
    0,
    SumAllReduce,
    hidden,
    hidden,
    topology.world_size(),
  )
  for layer in 0..<layer_count {
    let base = 1 + layer * 9
    llama_tp_append_collective(
      output,
      base + 4,
      SumAllReduce,
      hidden,
      hidden,
      topology.world_size(),
    )
    llama_tp_append_collective(
      output,
      base + 7,
      SumAllReduce,
      hidden,
      hidden,
      topology.world_size(),
    )
  }
  llama_tp_append_collective(
    output,
    2 + layer_count * 9,
    LastAxisAllGather,
    vocabulary,
    vocabulary / topology.world_size(),
    topology.world_size(),
  )
  ReadOnlyArray::from_array(output)
}

///|
fn llama_tp_build_operations(
  model : @plan.ModelPlan,
  topology : @tensor_parallel_plan.TensorParallelTopology,
) -> ReadOnlyArray[@tensor_parallel_plan.TensorParallelOperationPlan] raise LlamaTensorParallelError {
  let shape = model.shape_constraints()
  let world = topology.world_size()
  let hidden = shape.hidden_size()
  let local_hidden = hidden / world
  let local_intermediate = shape.intermediate_size() / world
  let local_vocabulary = shape.vocabulary_size() / world
  let local_query_heads = shape.attention_head_count() / world
  let local_kv_heads = shape.kv_head_count() / world
  let local_qkv_width = (local_query_heads + 2 * local_kv_heads) *
    shape.head_dimension()
  let output : Array[@tensor_parallel_plan.TensorParallelOperationPlan] = []
  for operation in model.operations() {
    let (
      input_width,
      output_width,
      inner_width,
      query_heads,
      kv_heads,
      head_dimension,
    ) = match operation.kind() {
      Embedding => (1, hidden, 0, 0, 0, 0)
      RmsNorm | ResidualAdd => (hidden, hidden, 0, 0, 0, 0)
      GatedMlp => (hidden, hidden, local_intermediate, 0, 0, 0)
      QkvProjection =>
        (
          hidden,
          local_qkv_width,
          0,
          local_query_heads,
          local_kv_heads,
          shape.head_dimension(),
        )
      RotaryEmbedding =>
        (
          local_qkv_width,
          local_qkv_width,
          0,
          local_query_heads,
          local_kv_heads,
          shape.head_dimension(),
        )
      CausalAttention =>
        (
          local_qkv_width,
          local_hidden,
          0,
          local_query_heads,
          local_kv_heads,
          shape.head_dimension(),
        )
      OutputProjection => (local_hidden, hidden, 0, 0, 0, 0)
      LanguageModelHead => (hidden, local_vocabulary, 0, 0, 0, 0)
      QkRmsNorm => raise UnsupportedPlan(operation=operation.id().as_int())
    }
    output.push(
      @tensor_parallel_plan.TensorParallelOperationPlan::new(
        operation=operation.id(),
        kind=operation.kind(),
        capability=operation.required_capability(),
        input_width~,
        output_width~,
        inner_width~,
        query_head_count=query_heads,
        kv_head_count=kv_heads,
        head_dimension~,
      ) catch {
        InvalidTensorParallelPlan(..) as error =>
          raise InvalidPhysicalPlan(error)
      },
    )
  }
  ReadOnlyArray::from_array(output)
}

///|
/// Derives one immutable local-rank weight and collective plan from the exact
/// dense-Llama semantic graph. Source offsets are relative to the already
/// authenticated safetensors payload; a loader adds the admitted payload base
/// and copies each recipe segment directly into its final local arena region.
fn llama_tp_wrap_rank_plan(
  body : () -> @tensor_parallel_plan.TensorParallelRankPlan raise @tensor_parallel_plan.TensorParallelPlanError,
) -> @tensor_parallel_plan.TensorParallelRankPlan raise LlamaTensorParallelError {
  body() catch {
    error => raise InvalidPhysicalPlan(error)
  }
}

///|
pub fn build_rank_plan(
  model : @plan.ModelPlan,
  bindings : @llama_weights.LlamaWeightBindings,
  rank~ : Int,
  world_size~ : Int,
  alignment_bytes~ : Int64,
  max_arena_bytes~ : Int64,
) -> @tensor_parallel_plan.TensorParallelRankPlan raise LlamaTensorParallelError {
  let topology = @tensor_parallel_plan.TensorParallelTopology::new(
    rank~,
    world_size~,
  ) catch {
    _ => raise InvalidTopology
  }
  if !llama_tp_valid_alignment(alignment_bytes) {
    raise InvalidAlignment
  }
  if max_arena_bytes <= 0L {
    raise ArenaTooLarge
  }
  let layer_count = llama_tp_validate_dense_plan(model, bindings)
  let shape = model.shape_constraints()
  let hidden = shape.hidden_size()
  let intermediate = shape.intermediate_size()
  let vocabulary = shape.vocabulary_size()
  let attention_heads = shape.attention_head_count()
  let kv_heads = shape.kv_head_count()
  llama_tp_require_divisible(vocabulary, world_size, Vocabulary)
  llama_tp_require_divisible(hidden, world_size, Hidden)
  llama_tp_require_divisible(intermediate, world_size, Intermediate)
  llama_tp_require_divisible(attention_heads, world_size, AttentionHeads)
  llama_tp_require_divisible(kv_heads, world_size, KeyValueHeads)
  let kv_width = kv_heads * shape.head_dimension()
  let tensors : Array[@tensor_parallel_plan.TensorRankPlacement] = []
  let arena_end = Ref(0L)
  let materialized = Ref(0L)
  llama_tp_append_tensor(
    tensors,
    bindings,
    topology,
    0,
    ShardRows,
    vocabulary.to_int64(),
    hidden.to_int64(),
    alignment_bytes,
    arena_end,
    materialized,
    max_arena_bytes,
  )
  for layer in 0..<layer_count {
    let base = 1 + layer * 9
    llama_tp_append_tensor(
      tensors,
      bindings,
      topology,
      base,
      Replicated,
      1L,
      hidden.to_int64(),
      alignment_bytes,
      arena_end,
      materialized,
      max_arena_bytes,
    )
    llama_tp_append_tensor(
      tensors,
      bindings,
      topology,
      base + 1,
      ShardRows,
      hidden.to_int64(),
      hidden.to_int64(),
      alignment_bytes,
      arena_end,
      materialized,
      max_arena_bytes,
    )
    for offset in 2..<=3 {
      llama_tp_append_tensor(
        tensors,
        bindings,
        topology,
        base + offset,
        ShardRows,
        kv_width.to_int64(),
        hidden.to_int64(),
        alignment_bytes,
        arena_end,
        materialized,
        max_arena_bytes,
      )
    }
    llama_tp_append_tensor(
      tensors,
      bindings,
      topology,
      base + 4,
      ShardColumns,
      hidden.to_int64(),
      hidden.to_int64(),
      alignment_bytes,
      arena_end,
      materialized,
      max_arena_bytes,
    )
    llama_tp_append_tensor(
      tensors,
      bindings,
      topology,
      base + 5,
      Replicated,
      1L,
      hidden.to_int64(),
      alignment_bytes,
      arena_end,
      materialized,
      max_arena_bytes,
    )
    for offset in 6..<=7 {
      llama_tp_append_tensor(
        tensors,
        bindings,
        topology,
        base + offset,
        ShardRows,
        intermediate.to_int64(),
        hidden.to_int64(),
        alignment_bytes,
        arena_end,
        materialized,
        max_arena_bytes,
      )
    }
    llama_tp_append_tensor(
      tensors,
      bindings,
      topology,
      base + 8,
      ShardColumns,
      hidden.to_int64(),
      intermediate.to_int64(),
      alignment_bytes,
      arena_end,
      materialized,
      max_arena_bytes,
    )
  }
  let final_norm = 1 + layer_count * 9
  llama_tp_append_tensor(
    tensors,
    bindings,
    topology,
    final_norm,
    Replicated,
    1L,
    hidden.to_int64(),
    alignment_bytes,
    arena_end,
    materialized,
    max_arena_bytes,
  )
  llama_tp_append_tensor(
    tensors,
    bindings,
    topology,
    final_norm + 1,
    ShardRows,
    vocabulary.to_int64(),
    hidden.to_int64(),
    alignment_bytes,
    arena_end,
    materialized,
    max_arena_bytes,
  )
  let collectives = llama_tp_build_collectives(
    layer_count, hidden, vocabulary, topology,
  )
  let operations = llama_tp_build_operations(model, topology)
  llama_tp_wrap_rank_plan(() => {
    @tensor_parallel_plan.TensorParallelRankPlan::new(
      model~,
      topology~,
      alignment_bytes~,
      arena_bytes=arena_end.val,
      materialized_bytes=materialized.val,
      tensors~,
      operations~,
      collectives~,
    )
  })
}
