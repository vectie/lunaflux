///|
test "row reuse envelope preserves wide staging and numerical law" {
  let policy = MatrixRowReuse::new(
    minimum_rows=32,
    minimum_inner=1536,
    wide_inner=8192,
    wide_tile=64,
  )
  for
    (rows, n, k, expected) in [
      (64, 4096, 8192, (32, 32, 64)),
      (64, 8192, 1536, (32, 32, 32)),
      (64, 16384, 512, (16, 64, 32)),
      (16, 4096, 8192, (16, 64, 32)),
      (64, 1536, 4096, (16, 64, 128)),
    ] {
    let plan = MatrixTilePlan::new(
      rows~,
      columns=n,
      inner=k,
      law=ShapeAwareFloat32,
      staging=WorksetBoundedTiles(
        inner_tile=128,
        maximum_weight_bytes=16777216L,
        maximum_tiles=192,
      ),
    )
    let selected = policy.apply(plan)
    assert_eq(selected.tile(), expected)
    assert_eq(selected.law(), plan.law())
    assert_eq(selected.geometry(), plan.geometry())
  }
  assert_true(
    (try? MatrixRowReuse::new(
      minimum_rows=0,
      minimum_inner=1,
      wide_inner=1,
      wide_tile=64,
    ))
    is Err(_),
  )
}
