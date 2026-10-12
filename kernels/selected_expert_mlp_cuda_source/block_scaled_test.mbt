///|
test "block scaled matrix lowering retains independent raw accumulator and launch ownership" {
  let law : @precision_ir.ScaledDotReductionLaw = UnscaledBlockMatrixFloat32(
    block_columns=128,
    publication=SeparateMultiplyAdd,
  )
  let plan = dynamic_plan(shared=true, scaled_dot_law=law)
  assert_eq(plan.scaled_dot_law(), law)
  let selected = @selected_expert_mlp_cuda_source.packed_expert_mlp_source(
    plan,
    symbol="matrix_shared",
  )
  assert_eq(selected.launches()[1], (8, 128))
  assert_eq(selected.launches()[3], (8, 128))
  let text = @utf8.decode(selected.source_bytes())
  assert_true(
    text.contains("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"),
  )
  assert_true(text.contains("float d[4]={0.0f,0.0f,0.0f,0.0f}"))
  assert_true(
    text.contains("total[i]=__fadd_rn(total[i], __fmul_rn(d[i], scale))"),
  )
  assert_true(text.contains("v=__fmul_rn(v, selected_weights[job]);"))
  assert_false(text.contains("matrix_shared_a0"))
  assert_false(text.contains("DeepSeek"))
  let ordinary = @selected_expert_mlp_cuda_source.packed_expert_mlp_source(
    dynamic_plan(shared=true),
    symbol="matrix_shared",
  )
  let explicit_default = @selected_expert_mlp_cuda_source.packed_expert_mlp_source(
    dynamic_plan(shared=true, scaled_dot_law=DecodeOperandsFloat32),
    symbol="matrix_shared",
  )
  assert_eq(ordinary.source_bytes(), explicit_default.source_bytes())
  assert_eq(plan.total_bytes(), ordinary.plan().total_bytes())
}

///|
test "unsupported block scaled formats do not silently fall back" {
  try
    @selected_expert_mlp_cuda_source.packed_expert_mlp_source(
      dynamic_plan(
        scaled_dot_law=UnscaledBlockMatrixFloat32(
          block_columns=128,
          publication=SeparateMultiplyAdd,
        ),
      ),
      symbol="bad",
    )
  catch {
    @luna_cuda_precision.LoweringError::UnsupportedGeometry => ()
    _ => fail("wrong unsupported scaled-dot error")
  } noraise {
    _ => fail("FP4 route silently used E4M3 matrix lowering")
  }
  try
    @selected_expert_mlp_cuda_source.packed_expert_mlp_source(
      dynamic_plan(
        shared=true,
        scaled_dot_law=UnscaledBlockMatrixFloat32(
          block_columns=64,
          publication=SeparateMultiplyAdd,
        ),
      ),
      symbol="bad",
    )
  catch {
    @luna_cuda_precision.LoweringError::UnsupportedGeometry => ()
    _ => fail("wrong unsupported scaled-dot error")
  } noraise {
    _ => fail("logical block changed silently")
  }
}
