///|
test "activation factorization changes each operand independently" {
  let cases = activation_combinations()
  assert_eq(cases.length(), 4)
  assert_eq(cases[0], ("first-input", 0, 0))
  assert_eq(cases[1], ("second-input", 1, 1))
  assert_eq(cases[2], ("gate-first-up-second", 0, 1))
  assert_eq(cases[3], ("gate-second-up-first", 1, 0))
  assert_eq(activation_geometry(140, 25600), 7168000L)
  for (rows, width) in [(0, 25600), (1025, 25600), (140, 0), (140, 65537)] {
    try activation_geometry(rows, width) catch {
      _ => ()
    } noraise {
      _ => fail("invalid activation geometry accepted")
    }
  }
}
