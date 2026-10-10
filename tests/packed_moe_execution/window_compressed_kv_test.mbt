///|
fn joint_attention_fixture(
  read_set : @precision.CompressedRowReadSet,
) -> @joint_frame.WindowCompressedKvFrame raise {
  @joint_frame.WindowCompressedKvFrame::new(
    @joint_source.window_compressed_kv_source(
      @precision.WindowCompressedKvPrecision::new(
        @precision.WindowSharedKvPrecision::new(
          query_heads=2,
          head_width=128,
          window_tokens=4,
          maximum_rows=3,
          maximum_positions=16,
          maximum_state_bytes=2580L,
        ),
        compression_ratio=4,
        read_set~,
      ),
      symbol_prefix="joint_fixture",
    ),
  )
}

///|
test "joint attention prepares four allocation-free effects for both read sets" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  let stream = context.create_stream()
  let aot = context.load_module(b"fake\x00")
  let operand = context.allocate(65536L)
  for read_set in [@precision.CompressedRowReadSet::Learned(top_k=2), AllCausal] {
    let frame = joint_attention_fixture(read_set)
    frame.prepare(
      context,
      aot,
      counts=operand,
      positions=operand,
      reset_flag=operand,
      queries=operand,
      shared_kv=operand,
      attention_sink=operand,
      compressed_values=operand,
      compressed_counts=operand,
      compressed_append=operand,
      read_ports=match read_set {
        Learned(top_k=_) =>
          @joint_frame.CompressedReadPorts::Learned(
            counts=operand,
            rows=operand,
            offsets=operand,
          )
        AllCausal => @joint_frame.CompressedReadPorts::AllCausal
      },
      maximum_state_bytes=2580L,
    )
    assert_eq(frame.launches().length(), 4)
    assert_eq(frame.output().byte_count(), 1536L)
    assert_eq(frame.append_descriptor().byte_count(), 12L)
    let queue = context.create_ordered_kernel_executor(stream, frame.launches())
    let before = enqueues()
    let waits = blocking_syncs()
    begin_allocations()
    for _ in 0..<32 {
      for stage in 0..<4 {
        queue.enqueue(stage)
      }
      queue.record_completion()
      while !queue.poll_completion() {
        ()
      }
      queue.reset()
    }
    assert_eq(end_allocations(), 0UL)
    assert_eq(enqueues() - before, 128UL)
    assert_eq(blocking_syncs(), waits)
    queue.close()
    frame.close()
    frame.close()
  }
  operand.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
test "joint attention rejects incompatible read ports or short cache before allocation" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  let aot = context.load_module(b"fake\x00")
  let operand = context.allocate(65536L)
  let short = context.allocate(1023L)
  let short_ids = context.allocate(23L)
  for case in 0..<3 {
    let frame = joint_attention_fixture(Learned(top_k=2))
    let before = live_resources()
    try
      frame.prepare(
        context,
        aot,
        counts=operand,
        positions=operand,
        reset_flag=operand,
        queries=operand,
        shared_kv=operand,
        attention_sink=operand,
        compressed_values=if case == 0 { short } else { operand },
        compressed_counts=operand,
        compressed_append=operand,
        read_ports=if case == 1 {
          @joint_frame.CompressedReadPorts::AllCausal
        } else {
          @joint_frame.CompressedReadPorts::Learned(
            counts=operand,
            rows=if case == 2 { short_ids } else { operand },
            offsets=operand,
          )
        },
        maximum_state_bytes=2580L,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid borrowed ports accepted")
    }
    assert_eq(live_resources(), before)
    frame.close()
  }
  short_ids.close()
  short.close()
  operand.close()
  aot.close()
  context.close()
  assert_eq(live_resources(), baseline)
}
