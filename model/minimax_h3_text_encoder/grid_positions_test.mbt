///|
test "video frames use timestamp text and reset temporal coordinate" {
  let c = parse_config(config_fixture(), max_bytes=4096)
  let ids = [
    10, 151652, 151656, 151656, 151653, 11, 151652, 151656, 151656, 151653,
  ]
  let p = c.placement_from_grids(ids, [
    VisualGrid::new(kind=Video, time=2, height=2, width=4),
  ])
  assert_eq(p.token_positions, ReadOnlyArray::from_array([2, 3, 7, 8]))
  assert_eq(p.rotary_positions[2], (2, 2, 2))
  assert_eq(p.rotary_positions[3], (2, 2, 3))
  assert_eq(p.rotary_positions[4], (4, 4, 4))
  assert_eq(p.rotary_positions[7], (7, 7, 7))
  assert_eq(p.rotary_positions[8], (7, 7, 8))
  assert_eq(p.rotary_positions[9], (9, 9, 9))
}

///|
test "image coordinates advance following text by max dimension not pixel count" {
  let c = parse_config(config_fixture(), max_bytes=4096)
  let ids = [151652, 151655, 151655, 151655, 151655, 151653, 2]
  let p = c.placement_from_grids(ids, [
    VisualGrid::new(kind=Image, time=1, height=4, width=4),
  ])
  assert_eq(p.rotary_positions[1], (1, 1, 1))
  assert_eq(p.rotary_positions[4], (1, 2, 2))
  assert_eq(p.rotary_positions[5], (3, 3, 3))
  assert_eq(p.rotary_positions[6], (4, 4, 4))
  try
    c.placement_from_grids(ids, [
      VisualGrid::new(kind=Video, time=1, height=4, width=4),
    ])
  catch {
    InvalidRequest => ()
    _ => fail("wrong media mismatch")
  } noraise {
    _ => fail("wrong grid admitted")
  }
}
