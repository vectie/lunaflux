///|
test "scalar physical lowering preserves separate rounding and ordered source" {
  let dot : CudaProjectionDot = {
    accumulator: "value",
    left: "x[inner]",
    right: "w[inner]",
  }
  assert_eq(
    projection_dot_fold_source(17, [dot], OrderedDot),
    "float value = 0.0f;\nfor (int inner = 0; inner < 17; inner += 1) {\n  value = __fadd_rn(value, __fmul_rn(x[inner], w[inner]));\n}\n",
  )
  let tree = projection_dot_fold_source(65, [dot], StridedPairwiseDot(32))
  assert_true(tree.contains("inner = lane; inner < 65; inner += 32"))
  assert_true(tree.contains("delta = 16; delta > 0; delta >>= 1"))
  assert_true(tree.contains("__fadd_rn(value, __fmul_rn("))
  assert_false(tree.contains("__fmaf"))
}

///|
test "panic scalar fold refuses conflicting shared operand bindings" {
  ignore(
    projection_dot_fold_source(
      16,
      [
        { accumulator: "x", left: "a[inner]", right: "b[inner]", },
        { accumulator: "y", left: "c[inner]", right: "d[inner]", },
      ],
      OrderedDot,
    ),
  )
}

///|
test "CUDA scalar join rejects generic worker geometries it cannot realize" {
  projection_validate_scalar_realization(OrderedDot)
  projection_validate_scalar_realization(StridedPairwiseDot(32))
  for workers in [1, 2, 16, 64, 128] {
    try
      projection_validate_scalar_realization(StridedPairwiseDot(workers))
    catch {
      Unsupported(Strategy) => ()
      _ => fail("wrong scalar realization failure")
    } noraise {
      _ => fail("unsupported worker geometry admitted")
    }
  }
}

///|
test "panic CUDA scalar emitter cannot skip half the domain with 64 workers" {
  ignore(
    projection_dot_fold_source(
      128,
      [{ accumulator: "value", left: "x[inner]", right: "w[inner]", }],
      StridedPairwiseDot(64),
    ),
  )
}
