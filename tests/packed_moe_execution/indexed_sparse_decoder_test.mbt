///|
test "complete sparse decoder and routed FFN share thirty-seven prepared launches" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let attention = @precision.IndexedSparseAttentionPrecision::new(
    index_head_count=2,
    index_head_dimension=4,
    query_head_count=2,
    key_value_head_count=2,
    query_key_dimension=4,
    value_dimension=4,
    pool_size=4,
    top_k=8,
    append_incomplete_tail=true,
    maximum_queries=2,
    maximum_history=32,
    maximum_workspace_bytes=100000L,
  )
  let projection = @precision.IndexedSparseProjectionPrecision::new(
    attention,
    hidden_width=32,
    query_rank=4,
    key_value_rank=2,
    rms_epsilon=1.0e-5,
    index_layer_norm_epsilon=1.0e-5,
    maximum_workspace_bytes=10000L,
  )
  let envelope = @precision.HyperConnectionPrecision::new(
    stream_count=4,
    hidden_width=32,
    maximum_rows=2,
    sinkhorn_iterations=20,
    control_epsilon=1.0e-6,
    normalization_epsilon=1.0e-5,
    mix_law=RoundedBf16ControlsAndProducts,
  )
  let decoder = @projection_frame.IndexedSparseDecoderFrame::new(
    @projection_source.indexed_sparse_decoder_source(
      @precision.IndexedSparseDecoderPrecision::new(
        projection,
        envelope,
        maximum_state_bytes=10000L,
        maximum_workspace_bytes=100000L,
      ),
      symbol_prefix="full_sparse",
    ),
  )
  defer (decoder.close() catch { _ => () })
  let ffn = @hyper_frame.HyperConnectionFrame::new(
    @hyper_source.hyper_connection_source(envelope, symbol_prefix="full_ffn"),
  )
  defer (ffn.close() catch { _ => () })
  let (moe, routed_bytes, shared_bytes) = routed_program_fixture(
    complete_local=true,
  )
  defer (moe.close() catch { _ => () })
  let external = [8192L, 4096L, 4096L, 4096L, routed_bytes, shared_bytes].map(bytes => {
    context.allocate(bytes)
  })
  defer (for buffer in external.rev() {
    buffer.close() catch {
      _ => ()
    }
  })
  ffn.prepare(
    context,
    aot,
    counts=external[0],
    input=external[2],
    output=external[3],
    function_weight=external[0],
    base=external[0],
    scale=external[0],
    norm_weight=external[0],
    maximum_workspace_bytes=ffn.workspace_bytes(),
  )
  decoder.prepare(
    context,
    aot,
    maximum_workspace_bytes=decoder.workspace_bytes(),
    maximum_state_bytes=decoder.state_bytes(),
    counts=external[0],
    reset_flag=external[0],
    hidden=external[1],
    output=external[2],
    projection_weights=Array::make(12, external[0]),
    pool_ape=external[0],
    output_weight=external[0],
    hyper_weights=Array::make(4, external[0]),
  )
  assert_eq(decoder.launches().length(), 23)
  moe.prepare_routed_frame(
    context,
    aot,
    maximum_workspace_bytes=moe.workspace_bytes(),
    counts=external[0],
    input=ffn.branch_input(),
    router_weight=external[0],
    correction_bias=external[0],
    routed_bank=external[4],
    shared_bank=external[5],
    output=ffn.branch_output(),
  )
  let launches = decoder.launches().to_owned()
  launches.append(ffn.prefix())
  launches.append(moe.launches())
  launches.append(ffn.suffix())
  assert_eq(launches.length(), 37)
  let queue = context.create_ordered_kernel_executor(stream, launches)
  defer (queue.abort() catch { _ => () })
  defer (queue.close() catch { _ => () })
  let before = enqueues()
  let waits = blocking_syncs()
  begin_allocations()
  for _ in 0..<32 {
    for stage in 0..<37 {
      queue.enqueue(stage)
    }
    queue.record_completion()
    while !queue.poll_completion() {
      ()
    }
    queue.reset()
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(enqueues() - before, 1184UL)
  assert_eq(blocking_syncs(), waits)
  queue.enqueue(0)
  queue.abort()
  queue.close()
  moe.close()
  ffn.close()
  decoder.close()
  decoder.close()
  for buffer in external.rev() {
    buffer.close()
  }
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}
