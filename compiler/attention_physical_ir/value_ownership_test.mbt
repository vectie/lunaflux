///|
test "value ownership is bijective for full and ragged dimensions" {
  for dimensions in [1, 31, 32, 33, 64, 96, 128, 129, 256] {
    let plan = ValueComponentOwnership::new(
      dimensions~,
      owners=32,
      maximum_fragment=4,
    ).unwrap()
    let visited = Array::make(dimensions, 0)
    for owner in 0..<32 {
      for item in 0..<((dimensions - 1) / 32 + 1) {
        if plan.component(owner, item) is Some(component) {
          visited[component] += 1
        }
      }
    }
    assert_true(visited.all(n => n == 1))
    if dimensions == 128 {
      assert_eq(plan.fragment_width(), 4)
    }
  }
  assert_eq(
    ValueComponentOwnership::new(dimensions=0, owners=32, maximum_fragment=4),
    None,
  )
}
