///|
fn probe_int_arrays_equal(
  left : ReadOnlyArray[Int],
  right : ArrayView[Int],
) -> Bool {
  if left.length() != right.length() {
    return false
  }
  for index in 0..<left.length() {
    if left[index] != right[index] {
      return false
    }
  }
  true
}

///|
fn require_exact_fused_qualification_shapes(
  shapes : ReadOnlyArray[@fused_parallel_qualification.FusedQualificationShape],
) -> Unit raise {
  require_fused_probe(shapes.length() == 4, "qualification shape count changed")
  let names = [
    "single-origin", "single-page-tail", "cross-page-pair", "eight-token-page-boundaries",
  ]
  let positions = [[0], [15], [15, 16], [0, 1, 14, 15, 16, 17, 30, 31]]
  let pages = [[1], [2], [1, 3], [0, 2]]
  for index in 0..<shapes.length() {
    let shape = shapes[index]
    require_fused_probe(shape.ordinal() == index, "shape ordinal changed")
    require_fused_probe(shape.name() == names[index], "shape name changed")
    require_fused_probe(
      probe_int_arrays_equal(shape.positions(), positions[index]),
      "shape positions changed",
    )
    require_fused_probe(
      probe_int_arrays_equal(shape.page_table(), pages[index]),
      "shape page table changed",
    )
    require_fused_probe(
      shape.token_count() == positions[index].length(),
      "shape token count changed",
    )
  }
}
