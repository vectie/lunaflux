///|
test "publication refinement reaches source without removing workgroup lifetime edges" {
  for
    (id, subgroup_publications, workgroup_publications) in [
      (450, 2, 3),
      (454, 1, 2),
      (460, 0, 4),
    ] {
    let lowering = source_test_decode_lowering(
      candidate_id=id,
      alternative=true,
    )
    let source = @utf8.decode(
      emit_cuda_attention_tile_source(
        lowering,
        source_test_abi(),
        "publication",
        1048576,
      ).bytes(),
    )
    let body = source.split("for (int key_base = key_begin;").to_array()[1]
      .split("if (!(denominator > 0.0f)")
      .to_array()[0]
    assert_eq(
      body.split("__syncwarp();").to_array().length(),
      subgroup_publications + 1,
    )
    // Terminal completion is outside the loop, and invalid exits retain
    // their own drain publication. Compare only the per-tile region.
    let per_tile = body.split("    // Terminal").to_array()[0]
    let normal = per_tile.split("__syncthreads(); return;").join("")
    assert_true(
      normal.split("__syncthreads();").to_array().length() >=
      workgroup_publications + 1,
    )
    assert_true(body.contains("float tile_maximum"))
    assert_true(body.contains("float tile_denominator"))
    assert_true(body.contains("float tile_numerator"))
    assert_true(body.contains("const float next_maximum"))
    if id == 460 {
      let phases = body
        .split("stage_key(key_base + 2 * LF_KEY_VALUE_TILE, slot);")
        .to_array()
      assert_eq(phases.length(), 2)
      assert_true(phases[0].contains("scores[key] = dot * scale"))
      assert_true(phases[0].contains("__syncthreads();"))
      assert_true(phases[1].contains("cp.async.wait_group 3;"))
      assert_true(phases[1].contains("__syncthreads();"))
    }
  }
}
