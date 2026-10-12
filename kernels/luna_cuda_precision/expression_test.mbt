///|
test "scalar lowering keeps explicit roundings scale order and literal bits" {
  let roundtrip = @precision.NumericExpression::new(
    [(BFloat16, FiniteFloat)],
    Convert(BFloat16, Convert(Float32, Input(0))),
  )
  assert_eq(@luna_cuda_precision.scalar_expression(roundtrip, ["x"]), "x")
  let expression = @precision.NumericExpression::new(
    [(Float32, AnyFloat), (BFloat16, AnyFloat), (Float32, AnyFloat)],
    Binary(
      AddRN,
      Binary(MultiplyRN, Input(0), Convert(Float32, Input(1))),
      Input(2),
    ),
  )
  assert_eq(
    @luna_cuda_precision.scalar_expression(expression, ["x", "w", "bias"]),
    "__fadd_rn(__fmul_rn(x, __bfloat162float(w)), bias)",
  )
  let fused = @precision.NumericExpression::new(
    expression.inputs()[:],
    FusedMultiplyAdd(Input(0), Convert(Float32, Input(1)), Input(2)),
  )
  assert_eq(
    @luna_cuda_precision.scalar_expression(fused, ["x", "w", "bias"]),
    "__fmaf_rn(x, __bfloat162float(w), bias)",
  )
  let constant = @precision.NumericExpression::new([], Float32Bits(0x80000000U))
  assert_eq(
    @luna_cuda_precision.scalar_expression(constant, []),
    "__uint_as_float(2147483648U)",
  )
  try @luna_cuda_precision.scalar_expression(expression, ["x"]) catch {
    @precision.IllegalPlacement => ()
    _ => fail("wrong error")
  } noraise {
    _ => fail("incorrect bindings accepted")
  }
}

///|
test "root and negation preserve distinct primitives and signed inputs" {
  let inverse = @precision.NumericExpression::new(
    [(Float32, AnyFloat)],
    Unary(ReciprocalSqrt, Input(0)),
  )
  let divided = @precision.NumericExpression::new(
    [(Float32, AnyFloat)],
    Binary(DivideRN, Float32Bits(0x3f800000U), Unary(SquareRoot, Input(0))),
  )
  assert_true(inverse.relation_to(divided) == DifferentProgram)
  assert_eq(
    @luna_cuda_precision.scalar_expression(divided, ["x"]),
    "__fdiv_rn(1.0f, sqrtf(x))",
  )
  let negated = @precision.NumericExpression::new(
    [(Float32, AnyFloat)],
    Unary(Negate, Input(0)),
  )
  assert_eq(@luna_cuda_precision.scalar_expression(negated, ["a+b"]), "-(a+b)")
}
