///|
fn main raise {
  let arguments = @env.args()
  let partitioned = arguments == [arguments[0], "partitioned"] ||
    arguments == [arguments[0], "partitioned-312"]
  let compile_query_rows = if arguments == [arguments[0], "partitioned-312"] {
    32
  } else if partitioned {
    64
  } else {
    128
  }
  let requested_candidate = match arguments {
    [_, "partitioned-312"] => Some(312)
    [_, "290"] => Some(290)
    [_, "291"] => Some(291)
    [_, "300"] => Some(300)
    [_, "310"] => Some(310)
    [_, "311"] => Some(311)
    [_, "312"] => Some(312)
    [_, "313"] => Some(313)
    [_, "314"] => Some(314)
    [_, "315"] => Some(315)
    _ => None
  }
  let shape = @luna_attention_strategy.AttentionShape::new(
    query_heads=16,
    kv_heads=4,
    head_dimension=128,
    tokens_per_page=16,
  )
  let problem = @luna_attention_strategy.AttentionCompileProblem::new(
    shape~,
    phase=Prefill,
    numeric_type=AttentionBFloat16,
    query_rows=compile_query_rows,
    context_tokens=4096,
    causal=true,
  )
  let capabilities = @luna_attention_strategy.AttentionCompilerCapabilities::new(
    subgroup_width=32,
    max_threads_per_block=256,
    max_shared_memory_bytes=100000L,
    max_pipeline_stages=2,
    supports_async_copy=false,
    matrix_shape=Some(
      @luna_attention_strategy.AttentionMatrixShape::new(
        rows=16,
        columns=16,
        reduction=16,
      ),
    ),
  )
  let backend = @luna_cuda_attention_autotune.cuda_sm120_rtx5060ti_backend()
  let target = @luna_cuda_attention_autotune.cuda_sm120_rtx5060ti_target()
  let records : ReadOnlyArray[@luna_attention_strategy.AttentionAutotuneRecord] = if partitioned &&
    requested_candidate is None {
    ReadOnlyArray::from_array([])
  } else {
    match requested_candidate {
      None =>
        @luna_cuda_attention_autotune.cuda_sm120_rtx5060ti_prefill_records(
          problem, capabilities,
        )
      Some(stable_id) => {
        let candidates = @luna_attention_strategy.generate_attention_tile_candidates(
          problem, capabilities,
        )
        guard candidates.filter(candidate => candidate.stable_id() == stable_id)
          is [candidate] else {
          raise Failure("requested attention tile candidate is unavailable")
        }
        [
          @luna_attention_strategy.AttentionAutotuneRecord::new(
            key=@luna_attention_strategy.AttentionAutotuneKey::new(
              backend~,
              target~,
              problem~,
              candidate~,
            ),
            median_latency_ns=1L,
            sample_count=3,
          ),
        ]
      }
    }
  }
  let device_target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let abi = @luna_cuda_attention_tile_source.CudaPagedAttentionTileAbi::new(
    query_heads=16,
    key_value_heads=4,
    head_dimension=128,
    tokens_per_page=16,
    input_row_width=3072,
    page_stride_values=8192L,
    physical_page_capacity=256L,
    maximum_query_rows=32,
    maximum_query_tokens=128,
    maximum_page_entries=256,
  )
  if partitioned {
    let compiled = @luna_cuda_attention_tile_aot.compile_cuda_partitioned_attention_tile_aot(
      @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotRequest::new(
        backend~,
        target_id=target,
        problem~,
        capabilities~,
        device_target~,
        maximum_shared_memory_bytes=100000L,
        abi~,
        partial_symbol="lunaflux_attention_prefill_partitioned_partial_v1",
        merge_symbol="lunaflux_attention_prefill_partitioned_merge_v1",
        maximum_workspace_bytes=67108864L,
        maximum_source_bytes=1048576,
      ),
      records,
    )
    println(@utf8.decode(compiled.source().bytes()))
  } else {
    let compiled = @luna_cuda_attention_tile_aot.compile_cuda_attention_tile_aot(
      @luna_cuda_attention_tile_aot.CudaAttentionTileAotRequest::new(
        backend~,
        target_id=target,
        problem~,
        capabilities~,
        device_target~,
        maximum_shared_memory_bytes=100000L,
        abi~,
        symbol="lunaflux_attention_prefill_tile_v1",
        maximum_source_bytes=1048576,
      ),
      records,
    )
    println(@utf8.decode(compiled.source().bytes()))
  }
}
