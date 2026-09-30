///|
test "blockwise lowering keeps terminal publication outside the KV loop" {
  for id in [450, 451, 452, 453, 454, 455, 456, 457] {
    let lowering = source_test_decode_lowering(
      candidate_id=id,
      alternative=true,
    )
    guard lowering.physical_program() is BlockwiseKeyProgram(_, effects) else {
      fail("missing blockwise program")
    }
    let text = @utf8.decode(
      emit_cuda_attention_tile_source(
        lowering,
        source_test_abi(),
        "pipeline",
        1048576,
      ).bytes(),
    )
    let loop_and_tail = text.split("for (int key_base = key_begin;").to_array()
    assert_eq(loop_and_tail.length(), 2)
    let sections = loop_and_tail[1].split("\n  }\n").to_array()
    let tile_body = sections[0]
    let tail = sections[1]
    // Acquire K and V publish separately. Synchronous transfer additionally
    // releases its single slot each epoch; asynchronous release is terminal.
    assert_eq(
      tile_body.split("__syncthreads();").to_array().length() - 1,
      if effects.is_pipelined() {
        2
      } else {
        3
      },
    )
    assert_eq(tail.has_prefix("  __syncthreads();\n"), effects.is_pipelined())
  }
}
