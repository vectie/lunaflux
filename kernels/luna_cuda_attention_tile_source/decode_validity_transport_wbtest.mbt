///|
test "register validity requires complete immutable page ownership" {
  let abi = source_test_abi()
  assert_true(
    decode_independent_validity_transport(abi, true, 32, 32) is Some(_),
  )
  assert_eq(decode_independent_validity_transport(abi, false, 32, 32), None)
  assert_eq(decode_independent_validity_transport(abi, true, 32, 1), None)
  assert_eq(decode_independent_validity_transport(abi, true, 8, 32), None)
  assert_eq(decode_independent_validity_transport(abi, true, 1024, 1024), None)
}

///|
test "register validity does not weaken readiness or invalid-stage drain" {
  guard source_test_decode_lowering(candidate_id=460, alternative=true).physical_program()
    is BlockwiseKeyProgram(_, effects) else {
    fail("missing effects")
  }
  for
    point in [
      @luna_attention_tile_schedule.AcquireKeyOperands,
      AcquireValueOperands,
    ] {
    let shared = cuda_grouped_effects(effects, point, "", "")
    let retained = cuda_grouped_effects(
      effects,
      point,
      "",
      "",
      retained_validity=true,
    )
    for text in [shared, retained] {
      assert_true(text.contains("cp.async.wait_group 3;"))
      assert_eq(text.split("__syncthreads();").length(), 3)
      assert_true(
        text.contains(
          "cp.async.wait_group 0;\" ::: \"memory\"); __syncthreads(); return;",
        ),
      )
    }
    assert_true(shared.contains("stage_valid["))
    assert_false(retained.contains("stage_valid["))
  }
}

///|
test "independent value validity snapshots precede early future key refill" {
  let lowering = source_test_decode_lowering(candidate_id=468, alternative=true)
  let text = @utf8.decode(
    emit_cuda_attention_tile_source(
      lowering,
      source_test_abi(),
      "validity_retention",
      1048576,
    ).bytes(),
  )
  assert_false(text.contains("int *tile_valid"))
  assert_false(text.contains("int *stage_valid"))
  assert_false(text.contains("atomicExch"))
  assert_true(text.contains("value_stage_valid_0 = operand_valid"))
  let phases = text
    .split("stage_key(key_base + 2 * LF_KEY_VALUE_TILE, slot);")
    .to_array()
  assert_eq(phases.length(), 2)
  assert_true(
    phases[1].contains("slot == 0 ? value_stage_valid_0 : value_stage_valid_1"),
  )
  assert_false(phases[1].contains("if (!(slot == 0 ? future_valid_0"))
  let value_phase = phases[1]
    .split("if (!(slot == 0 ? value_stage_valid_0 : value_stage_valid_1))")
    .to_array()
  assert_eq(value_phase.length(), 2)
  assert_true(value_phase[0].contains("cp.async.wait_group 3;"))
  assert_false(
    value_phase[0].contains("reinterpret_cast<const uint2 *>(value_tile"),
  )
  assert_true(
    value_phase[1].contains("reinterpret_cast<const uint2 *>(value_tile"),
  )
}

///|
test "page-validity ballots are uniform before predicated vector copies" {
  let lowering = source_test_decode_lowering(candidate_id=468, alternative=true)
  guard lowering.physical_program() is BlockwiseKeyProgram(_, effects) else {
    fail("missing effects")
  }
  let source = emit_decode_async_stage(
    source_test_abi(),
    effects,
    true,
    lowering.decode_copy_retention().unwrap(),
    vector_elements=8,
    tile_rows=32,
    start_alignment=32,
  )
  for operand in ["key", "value"] {
    let body = source.split("auto stage_" + operand).to_array()[1]
    let prior = body.split("const int operand_valid = __all_sync").to_array()[0]
    assert_false(prior.contains("if (vector"))
    assert_false(prior.contains("if (active_keys"))
    assert_true(body.contains("lane * 16 >= active_keys || page_valid"))
  }
  assert_true(
    source.contains("stage_key(key_begin, 0); stage_value(key_begin, 0);"),
  )
  assert_false(source.contains("if (key_begin < key_limit) stage_key"))
}
