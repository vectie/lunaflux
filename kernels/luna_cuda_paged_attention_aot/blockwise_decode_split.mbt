///|
/// Exact executable ABI geometry for the blockwise F32 partition/merge law.
/// The partition count is an explicit AOT input, not inferred from latency or
/// a model name. Both source production and startup binding consume this plan.
pub fn blockwise_decode_split_launch_geometry(
  shape : @luna_attention_strategy.AttentionShape,
  splits : Int,
  candidate_id? : Int = 452,
) -> (Int, Int, Int, Int64) raise PagedAttentionAotError {
  let (_, _, candidate) = blockwise_decode_split_candidate(
    shape, 1, splits, candidate_id,
  )
  blockwise_decode_execution_launch_geometry(
    shape,
    splits,
    candidate.compute(),
    candidate.tile().dimensions().1,
    candidate.execution_geometry().1,
  )
}

///|
/// Startup binding of an already admitted execution descriptor. This does not
/// enumerate or select compiler candidates and has no model-specific policy.
pub fn blockwise_decode_execution_launch_geometry(
  shape : @luna_attention_strategy.AttentionShape,
  splits : Int,
  compute : @luna_attention_strategy.AttentionTileCompute,
  key_value_tile_rows : Int,
  pipeline_stages : Int,
) -> (Int, Int, Int, Int64) raise PagedAttentionAotError {
  let (heads, kv, d, _) = shape.dimensions()
  guard splits > 1 &&
    splits <= 64 &&
    kv > 0 &&
    heads % kv == 0 &&
    (key_value_tile_rows == 32 || key_value_tile_rows == 64) &&
    (pipeline_stages == 1 || pipeline_stages == 2) else {
    raise Unsupported(Shape)
  }
  let group = heads / kv
  let operand_bytes = 2L *
    key_value_tile_rows.to_int64() *
    pipeline_stages.to_int64() *
    d.to_int64() *
    2L
  match compute {
    AttentionGroupedQueryMatrixQkPv => {
      guard group > 0 && group <= 16 && (d == 64 || d == 128) else {
        raise Unsupported(Shape)
      }
      (1, kv, 32, 16L * d.to_int64() * 2L + operand_bytes + 16L)
    }
    AttentionGroupedQueryBlockwiseDot
    | AttentionGroupedQueryDualScoreBlockwiseDot
    | AttentionGroupedQueryFmaBlockwiseDot
    | AttentionGroupedQueryOwnedBlockwiseDot(..) => {
      guard group > 0 && group <= 8 && d > 0 && d <= 256 else {
        raise Unsupported(Shape)
      }
      if compute is AttentionGroupedQueryOwnedBlockwiseDot(score_owners~, ..) &&
        score_owners != 2 &&
        score_owners != 4 &&
        score_owners != 8 {
        raise Unsupported(Shape)
      }
      (
        1,
        kv,
        group * 32,
        operand_bytes +
        group.to_int64() * key_value_tile_rows.to_int64() * 4L +
        16L,
      )
    }
    _ => raise Unsupported(Shape)
  }
}

///|
/// Derive the closed terminal family and its storage from typed candidates,
/// not numeric IDs or a fixed KV32 launch copied into an exporter.
fn blockwise_decode_split_candidate(
  shape : @luna_attention_strategy.AttentionShape,
  context : Int,
  splits : Int,
  candidate_id : Int,
) -> (
  @luna_attention_strategy.AttentionCompileProblem,
  @luna_attention_strategy.AttentionCompilerCapabilities,
  @luna_attention_strategy.AttentionTileCandidate,
) raise PagedAttentionAotError {
  let (heads, kv, d, _) = shape.dimensions()
  guard splits > 1 && splits <= 64 && kv > 0 && heads % kv == 0 else {
    raise Unsupported(Shape)
  }
  let group = heads / kv
  // An envelope for the finite KV32/KV64, one/two-stage frontier. The selected
  // candidate's exact storage is used for both compilation and launch binding.
  let envelope = (16L + 2L * 64L * 2L) * d.to_int64() * 2L +
    group.to_int64() * 64L * 4L +
    16L
  let problem = @luna_attention_strategy.AttentionCompileProblem::new(
    shape~,
    phase=Decode,
    numeric_type=AttentionBFloat16,
    query_rows=1,
    batch_rows=1,
    context_tokens=context,
    causal=true,
    allow_alternative_softmax_reduction=true,
  ) catch {
    _ => raise Unsupported(Profile)
  }
  let capabilities = @luna_attention_strategy.AttentionCompilerCapabilities::new(
    subgroup_width=32,
    max_threads_per_block=256,
    max_shared_memory_bytes=envelope,
    max_pipeline_stages=2,
    supports_async_copy=true,
    parallel_workgroup_target=kv * splits,
    matrix_shape=Some(
      @luna_attention_strategy.AttentionMatrixShape::new(
        rows=16,
        columns=16,
        reduction=16,
      ) catch {
        _ => raise Unsupported(Profile)
      },
    ),
  ) catch {
    _ => raise Unsupported(Profile)
  }
  let candidates = @luna_attention_strategy.generate_attention_tile_candidates(
    problem, capabilities,
  ) catch {
    _ => raise Unsupported(Profile)
  }
  guard candidates.filter(candidate => candidate.stable_id() == candidate_id)
    is [candidate] else {
    raise Unsupported(Profile)
  }
  guard candidate.compute()
    is (AttentionGroupedQueryBlockwiseDot
    | AttentionGroupedQueryDualScoreBlockwiseDot
    | AttentionGroupedQueryFmaBlockwiseDot
    | AttentionGroupedQueryOwnedBlockwiseDot(..)
    | AttentionGroupedQueryMatrixQkPv) else {
    raise Unsupported(Profile)
  }
  (problem, capabilities, candidate)
}

///|
/// Offline adapter only. It specializes a legal compiler candidate through
/// semantic → ownership/effects → partition map/merge → device lowering.
/// Runtime preparation consumes only the geometry and module, never this API.
pub fn compile_blockwise_decode_split(
  shape : @luna_attention_strategy.AttentionShape,
  profile : @launch_contract.PagedKernelProfile,
  target : @catalog.DeviceTarget,
  page_stride_bytes : Int64,
  physical_page_capacity : Int64,
  maximum_context_tokens : Int,
  splits : Int,
  maximum_workspace_bytes : Int64,
  candidate_id? : Int = 452,
) -> @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotCompilation raise PagedAttentionAotError {
  let (problem, capabilities, candidate) = blockwise_decode_split_candidate(
    shape, maximum_context_tokens, splits, candidate_id,
  )
  let (heads, kv, dimension, page) = shape.dimensions()
  let geometry = blockwise_decode_split_launch_geometry(
    shape,
    splits,
    candidate_id~,
  )
  let compiled = @luna_cuda_attention_tile_aot.specialize_cuda_partitioned_attention_tile_aot(
    @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotRequest::new(
      backend=@luna_attention_strategy.AttentionBackendId::new("cuda"),
      target_id=@luna_attention_strategy.AttentionTargetId::new(
        "decode-split-candidate-\{candidate.stable_id()}",
      ),
      problem~,
      capabilities~,
      device_target=target,
      maximum_shared_memory_bytes=geometry.3,
      abi=@luna_cuda_attention_tile_source.CudaPagedAttentionTileAbi::new(
        query_heads=heads,
        key_value_heads=kv,
        head_dimension=dimension,
        tokens_per_page=page,
        input_row_width=(heads + 2 * kv) * dimension,
        page_stride_values=page_stride_bytes / 2L,
        physical_page_capacity~,
        maximum_query_rows=profile.max_query_rows(),
        maximum_query_tokens=profile.max_query_tokens(),
        maximum_page_entries=profile.max_page_table_entries(),
      ),
      partial_symbol="lunaflux_paged_attention_bf16_decode_split_partial_v3_ep_3903",
      merge_symbol="lunaflux_paged_attention_bf16_decode_split_merge_v3_ep_3904",
      maximum_workspace_bytes~,
      maximum_source_bytes=1048576,
    ),
    candidate_id,
  ) catch {
    _ => raise Unsupported(Profile)
  }
  if compiled.lowering().key_value_partitions() != splits ||
    compiled.lowering().launch() != geometry {
    raise Mismatch(Contract)
  }
  compiled
}
