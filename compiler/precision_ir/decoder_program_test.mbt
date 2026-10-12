///|
test "conditional programs are typed and no empirical equivalence is inferred" {
  let direct = @precision_ir.SigmoidEvaluationLaw::DirectReciprocal.expression()
  let stable = @precision_ir.SigmoidEvaluationLaw::StableBranches.expression()
  assert_true(direct.relation_to(stable) == DifferentProgram)
  try
    @precision_ir.NumericExpression::new(
      [(Float32, AnyFloat)],
      Select(
        GreaterThan,
        Input(0),
        Float32Bits(0U),
        Input(0),
        Convert(BFloat16, Input(0)),
      ),
    )
  catch {
    @precision_ir.IllegalPlacement => ()
    _ => fail("wrong error")
  } noraise {
    _ => fail("mixed arm types accepted")
  }
  let separate = @precision_ir.Float32AffineLaw::SeparateMultiplyAdd.expression()
  let fused = @precision_ir.Float32AffineLaw::FusedMultiplyAdd.expression()
  assert_true(separate.relation_to(fused) == DifferentProgram)
  assert_true(
    @precision_ir.ReciprocalRootLaw::DivideRoundedSqrtFloat32
    .expression()
    .relation_to(
      @precision_ir.ReciprocalRootLaw::ReciprocalSqrtFloat32.expression(),
    ) ==
    DifferentProgram,
  )
}

///|
test "dense projection folds and effects preserve materialized rounding boundary" {
  let plan = @precision_ir.DenseSwiGluPrecision::new(
    hidden_width=8,
    intermediate_width=12,
    maximum_rows=3,
  )
  assert_eq(plan.projection_fold(down=false).extent, 8)
  assert_eq(plan.projection_fold(down=true).extent, 12)
  assert_eq(plan.projection_fold(down=true).initial_bits, 0U)
  assert_true(plan.expression(Activation).output() == BFloat16)
  assert_true(plan.expression(Product).output() == BFloat16)
  let program = plan.program()
  assert_eq(program.steps().length(), 2)
  assert_eq(program.lower(Ordered).commands(ComputeLane).length(), 2)
}

///|
test "rotary contraction remains a different numerical program" {
  for imaginary in [false, true] {
    let separate = @precision_ir.RotaryProductLaw::SeparateFloat32Products.expression(
      imaginary~,
    )
    let fused = @precision_ir.RotaryProductLaw::ComplexFloat32Fma.expression(
      imaginary~,
    )
    assert_true(separate.output() == BFloat16)
    assert_true(separate.relation_to(fused) == DifferentProgram)
  }
}

///|
test "estimated divide cannot pass the exact rewrite gate" {
  let rn = @precision_ir.NumericExpression::new(
    [(Float32, AnyFloat), (Float32, AnyFloat)],
    Binary(DivideRN, Input(0), Input(1)),
  )
  let estimate = @precision_ir.NumericExpression::new(
    [(Float32, AnyFloat), (Float32, AnyFloat)],
    Binary(DivideEstimate, Input(0), Input(1)),
  )
  assert_true(rn.relation_to(estimate) == DifferentProgram)
  try rn.exact_replacement(estimate) catch {
    @precision_ir.MissingAccuracyContract => ()
    _ => fail("wrong error")
  } noraise {
    _ => fail("estimate silently substituted for rounded division")
  }
  assert_true(
    @precision_ir.SparseExpression::Probability.expression().output() ==
    BFloat16,
  )
}

///|
test "sparse effects derive pooling selection attention dependencies" {
  let plan = @precision_ir.IndexedSparseAttentionPrecision::new(
    index_head_count=2,
    index_head_dimension=4,
    query_head_count=4,
    key_value_head_count=2,
    query_key_dimension=4,
    value_dimension=6,
    pool_size=2,
    top_k=4,
    append_incomplete_tail=true,
    maximum_queries=3,
    maximum_history=8,
    maximum_workspace_bytes=4096L,
  )
  let program = plan.program()
  assert_eq(program.steps().length(), 3)
  assert_eq(program.lower(Ordered).commands(ComputeLane).length(), 3)
}
