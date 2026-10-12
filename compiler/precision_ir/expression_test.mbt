///|
fn expression(
  root : @precision_ir.NumericExpr,
) -> @precision_ir.NumericExpression raise {
  @precision_ir.NumericExpression::new(
    [(Float32, AnyFloat), (Float32, AnyFloat), (Float32, AnyFloat)],
    root,
  )
}

///|
test "finite precision structure rejects reassociation contraction and scale motion" {
  let a = expression(Binary(AddRN, Binary(AddRN, Input(0), Input(1)), Input(2)))
  let b = expression(Binary(AddRN, Input(0), Binary(AddRN, Input(1), Input(2))))
  assert_true(a.relation_to(b) == DifferentProgram)
  let separate = expression(
    Binary(AddRN, Binary(MultiplyRN, Input(0), Input(1)), Input(2)),
  )
  let fused = expression(FusedMultiplyAdd(Input(0), Input(1), Input(2)))
  assert_true(separate.relation_to(fused) == DifferentProgram)
  let scaled_before = expression(
    Binary(
      AddRN,
      Binary(MultiplyRN, Input(0), Input(2)),
      Binary(MultiplyRN, Input(1), Input(2)),
    ),
  )
  let scaled_after = expression(
    Binary(MultiplyRN, Binary(AddRN, Input(0), Input(1)), Input(2)),
  )
  assert_true(scaled_before.relation_to(scaled_after) == DifferentProgram)
  try separate.exact_replacement(fused) catch {
    @precision_ir.MissingAccuracyContract => ()
    _ => fail("wrong error")
  } noraise {
    _ => fail("empirical alternative authorized as exact")
  }
  try separate.bounded_replacement(fused, maximum_absolute_error=1.0e30) catch {
    @precision_ir.MissingAccuracyContract => ()
    _ => fail("wrong error")
  } noraise {
    _ => fail("a large allowance manufactured a missing theorem")
  }
}

///|
test "finite BF16 roundtrip is exact conditionally not for arbitrary NaN payloads" {
  for domain in [@precision_ir.FiniteFloat, BoundedFinite(1.0), AnyFloat] {
    let a = @precision_ir.NumericExpression::new(
      [(BFloat16, domain)],
      Convert(BFloat16, Convert(Float32, Input(0))),
    )
    let b = @precision_ir.NumericExpression::new([(BFloat16, domain)], Input(0))
    assert_true(
      a.relation_to(b) ==
      (if domain == AnyFloat {
        DifferentProgram
      } else {
        ExactFiniteBf16Roundtrip
      }),
    )
    if domain != AnyFloat {
      assert_true(a.exact_replacement(b).root() == Input(0))
    }
  }
  let any = @precision_ir.NumericExpression::new(
    [(BFloat16, AnyFloat)],
    Input(0),
  )
  let finite = @precision_ir.NumericExpression::new(
    [(BFloat16, FiniteFloat)],
    Input(0),
  )
  assert_true(any.relation_to(finite) == DifferentProgram)
  let nested = @precision_ir.NumericExpression::new(
    [(BFloat16, FiniteFloat)],
    Binary(
      MultiplyRN,
      Convert(Float32, Convert(BFloat16, Convert(Float32, Input(0)))),
      Float32Bits(0x3f800000U),
    ),
  )
  assert_true(
    nested.relation_to(nested.simplify_exact()) == ExactFiniteBf16Roundtrip,
  )
  assert_true(
    nested.simplify_exact().root() ==
    Binary(MultiplyRN, Convert(Float32, Input(0)), Float32Bits(0x3f800000U)),
  )
  let twice = @precision_ir.NumericExpression::new(
    [(BFloat16, FiniteFloat)],
    Convert(
      BFloat16,
      Convert(Float32, Convert(BFloat16, Convert(Float32, Input(0)))),
    ),
  )
  assert_true(twice.simplify_exact().root() == Input(0))
  assert_true(
    twice.simplify_exact().simplify_exact().root() ==
    twice.simplify_exact().root(),
  )
}

///|
test "local BF16 bound is analytic domain qualified and never an exact rewrite" {
  let a = @precision_ir.NumericExpression::new(
    [(Float32, BoundedFinite(8.0))],
    Input(0),
  )
  let b = @precision_ir.NumericExpression::new(
    [(Float32, BoundedFinite(8.0))],
    Convert(Float32, Convert(BFloat16, Input(0))),
  )
  guard a.relation_to(b) is BoundedBf16Rounding(absolute_error~) else {
    fail("missing bound")
  }
  assert_true(absolute_error >= 8.0 / 256.0 && absolute_error <= 0.063)
  assert_true(
    a.bounded_replacement(b, maximum_absolute_error=0.063).root() == b.root(),
  )
  try a.bounded_replacement(b, maximum_absolute_error=0.001) catch {
    @precision_ir.MissingAccuracyContract => ()
    _ => fail("wrong error")
  } noraise {
    _ => fail("insufficient budget accepted")
  }
  try a.exact_replacement(b) catch {
    @precision_ir.MissingAccuracyContract => ()
    _ => fail("wrong error")
  } noraise {
    _ => fail("bounded rewrite accepted as exact")
  }
  let unbounded = @precision_ir.NumericExpression::new(
    [(Float32, FiniteFloat)],
    Input(0),
  )
  let rounded = @precision_ir.NumericExpression::new(
    [(Float32, FiniteFloat)],
    b.root(),
  )
  assert_true(unbounded.relation_to(rounded) == DifferentProgram)
  let overflow = @precision_ir.NumericExpression::new(
    [(Float32, BoundedFinite(3.4e38))],
    Input(0),
  )
  let overflow_rounded = @precision_ir.NumericExpression::new(
    overflow.inputs()[:],
    b.root(),
  )
  assert_true(overflow.relation_to(overflow_rounded) == DifferentProgram)
}

///|
test "input types slots constants and storage roundings are checked" {
  for
    root in [
      @precision_ir.Input(1),
      Binary(AddRN, Input(0), Input(0)),
      FusedMultiplyAdd(Input(0), Input(0), Input(0)),
      Unary(NaturalExp, Input(0)),
    ] {
    try
      @precision_ir.NumericExpression::new([(BFloat16, AnyFloat)], root)
    catch {
      @precision_ir.IllegalPlacement => ()
      _ => fail("wrong error")
    } noraise {
      _ => fail("invalid typed expression accepted")
    }
  }
  let positive = expression(Float32Bits(0U))
  let negative = expression(Float32Bits(0x80000000U))
  assert_true(positive.relation_to(negative) == DifferentProgram)
  assert_true(positive.relation_to(positive) == IdenticalProgram)
  try
    @precision_ir.NumericExpression::new(
      [(Float32, BoundedFinite(-1.0))],
      Input(0),
    )
  catch {
    @precision_ir.IllegalPlacement => ()
    _ => fail("wrong error")
  } noraise {
    _ => fail("invalid domain")
  }
}
