///|
test "export compiler repair probes without changing production ABIs" {
  if @env.get_env_var("LUNA_TEST_EXPORT_COUNTER_REPAIR") != Some("1") {
    return
  }
  repair_export_retained_prefill()
  for id in [322, 2000, 2001, 2003, 2004] {
    let lowering = source_test_lowering(true, 100000L, 2, candidate_id=Some(id))
    let source = emit_cuda_attention_tile_source(
      lowering,
      source_test_abi(),
      "lunaflux_attention_prefill_tile_v1",
      1048576,
    )
    println("LUNA_SOURCE_BEGIN prefill-\{id}")
    println("#define LF_SHARED_MEMORY_BYTES \{lowering.launch().3}")
    println(@utf8.decode(source.bytes()))
    println("LUNA_SOURCE_END")
  }
  for id in [430, 440, 441, 450, 451, 452, 453, 454, 455] {
    let lowering = source_test_decode_lowering(
      candidate_id=id,
      alternative=id >= 450,
    )
    let source = emit_cuda_attention_tile_source(
      lowering,
      source_test_abi(),
      "owned_decode",
      1048576,
    )
    println("LUNA_SOURCE_BEGIN decode-\{id}")
    println("#define LF_PROBE_BLOCK_THREADS \{lowering.launch().2}")
    println("#define LF_SHARED_MEMORY_BYTES \{lowering.launch().3}")
    println(@utf8.decode(source.bytes()))
    if id >= 454 {
      println("#define owned_decode owned_decode_dual_score_blockwise_f32_v2")
    } else if id >= 450 {
      println("#define owned_decode owned_decode_blockwise_f32_v1")
    }
    println("LUNA_SOURCE_END")
  }
}
