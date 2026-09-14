///|
test "concatenated weight lowering preserves QKV selection and generalizes operands" {
  let segments = @luna_projection_tile_compiler.plan_projection_concatenated_rows([
      8, 2, 2,
    ],
  ).unwrap()
  let (binding, branch) = projection_concatenated_weight_selection(
    segments[:],
    ["query_weight", "key_weight", "value_weight"],
    "column",
    "weight",
    "local_column",
  )
  assert_eq(
    binding, "const __nv_bfloat16 *weight = query_weight; int local_column = column;",
  )
  assert_eq(
    branch, "if (local_column >= 8) { local_column -= 8; if (local_column >= 2) { local_column -= 2; weight = value_weight; } else weight = key_weight; }",
  )
  let single = @luna_projection_tile_compiler.plan_projection_concatenated_rows([
      17,
    ],
  ).unwrap()
  assert_eq(
    projection_concatenated_weight_selection(single[:], ["a"], "r", "p", "i"),
    ("const __nv_bfloat16 *p = a; int i = r;", ""),
  )
  let four = @luna_projection_tile_compiler.plan_projection_concatenated_rows([
    3, 5, 7, 11,
  ]).unwrap()
  let (_, four_branch) = projection_concatenated_weight_selection(
    four[:],
    ["a", "b", "c", "d"],
    "r",
    "p",
    "i",
  )
  assert_eq(
    four_branch, "if (i >= 3) { i -= 3; if (i >= 5) { i -= 5; if (i >= 7) { i -= 7; p = d; } else p = c; } else p = b; }",
  )
}
