///|
test "qualification shapes are independently pinned" {
  require_exact_fused_qualification_shapes(
    @fused_parallel_qualification.declared_qualification_shapes(),
  )
}

///|
test "probe integer and BF16 codecs are exact little endian" {
  inspect(
    probe_i32_bytes([1, -1]),
    content=(
      #|b"\x01\x00\x00\x00\xff\xff\xff\xff"
    ),
  )
  debug_inspect(probe_bf16_bits(b"\x80\x3f\x00\xc0"), content="[16256, 49152]")
  inspect(probe_u32(b"\x78\x56\x34\x12"), content="305419896")
}

///|
test "resource ledger starts closed" {
  let balance = fused_probe_resource_balance()
  assert_true(balance.is_zero())
  assert_eq(
    balance.render(),
    "context0,stream0,event0,graph0,graph_exec0,allocation0,module0,function0,executor0,device_bytes0,pending0,cleanup0",
  )
}

///|
test "probe derives exact authority-free production V2 candidates" {
  let (_, production) = fused_production_probe_candidates("b".repeat(64))
  assert_false(production.qkv.manifest_bindable())
  assert_false(production.residual.manifest_bindable())
  assert_false(production.readonly_candidate.manifest_bindable())
  assert_true(
    @utf8.decode(production.qkv.recipe_bytes()).contains(
      "abi=step_counts,query_positions,query_row_offsets",
    ),
  )
  assert_true(
    @utf8.decode(production.readonly_candidate.recipe_bytes()).contains(
      "diagnostic_canary=absent",
    ),
  )
  assert_true(
    @utf8.decode(production.residual.recipe_bytes()).contains(
      "diagnostic_canary=absent",
    ),
  )
}

///|
test "probe derives exact authority-free sm120 candidates" {
  let set = fused_probe_candidate_set("b".repeat(64))
  assert_false(set.manifest_bindable())
  assert_eq(set.qkv_candidate().target().compute_major(), 12)
  assert_eq(set.qkv_candidate().target().compute_minor(), 0)
  assert_eq(set.qkv_candidate().launch_dimensions().block_x(), 128)
  assert_eq(set.residual_norm_candidate().target().compute_major(), 12)
  assert_eq(set.residual_norm_candidate().target().compute_minor(), 0)
  assert_eq(set.residual_norm_candidate().launch_dimensions().block_x(), 128)
}
