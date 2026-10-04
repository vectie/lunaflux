///|
test "prefix row domain retires whole groups and retains partial final groups" {
  for groups in [1, 2, 4, 8] {
    let program = FragmentProgram::new(
      rows=groups,
      columns=2,
      extent=64,
      quantum=16,
      slots=2,
    ).unwrap()
    let domain = program.row_domain(quantum=16).unwrap()
    assert_eq(domain.geometry(), (groups, 16))
    for remaining in -1..<=(groups * 16 + 1) {
      let live = domain.active_groups(remaining)
      for row in 0..<groups {
        assert_eq(row < live, row * 16 < remaining)
      }
    }
    assert_eq(domain.active_groups(2147483647), groups)
    assert_eq(domain.active_groups(-2147483647 - 1), 0)
    assert_eq(program.row_domain(quantum=0), None)
    if groups > 1 {
      assert_eq(program.row_domain(quantum=2147483647), None)
    }
  }
}
