///|
test "fifty complete layers terminate at second residual, not final norm" {
  let recipe = layer_function_order()
  assert_eq(recipe.length(), 13)
  assert_eq(1 + 50 * recipe.length(), 651)
  assert_eq(recipe[0], 1)
  assert_eq(recipe[4], 4)
  assert_eq(recipe[5], 5)
  assert_eq(recipe[6], 6)
  assert_eq(recipe[8], 10)
  assert_eq(recipe[9], 1)
  assert_eq(recipe[12], 10)
}
