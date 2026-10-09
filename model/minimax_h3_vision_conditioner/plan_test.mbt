///|
test "vision merged groups preserve frame attention boundaries and spatial positions" {
  let p = VisionPlan::new(
    [{ time: 2, height: 4, width: 6, }, { time: 1, height: 2, width: 2, }],
    max_workspace_bytes=10000000L,
    max_output_bytes=10000000L,
  )
  assert_eq(p.rows, 52)
  assert_eq(p.merged_rows, 13)
  assert_eq(p.position(0), (0, 24, 0, 0))
  assert_eq(p.position(1), (0, 24, 0, 1))
  assert_eq(p.position(2), (0, 24, 1, 0))
  assert_eq(p.position(4), (0, 24, 0, 2))
  assert_eq(p.position(24), (24, 48, 0, 0))
  assert_eq(p.position(51), (48, 52, 1, 1))
  assert_eq(p.merged_position(7), (0, 1, 0, 1))
  assert_eq(p.merged_position(12), (1, 0, 0, 0))
  let regions = p.workspace_regions()
  for i in 1..<regions.length() {
    assert_eq(regions[i].0, regions[i - 1].0 + regions[i - 1].1)
  }
  assert_eq(regions[7].0 + regions[7].1, p.workspace_bytes)
  assert_eq(p.output_offset(3) + p.output_bytes / 4L, p.output_bytes)
}

///|
test "vision geometry and budgets reject partial merge groups" {
  let invalid : Array[Grid] = [
    { time: 1, height: 3, width: 4, },
    { time: 0, height: 4, width: 4, },
    { time: 2147483647, height: 4, width: 4, },
  ]
  for g in invalid {
    try
      VisionPlan::new(
        [g],
        max_workspace_bytes=1000000L,
        max_output_bytes=1000000L,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid grid accepted")
    }
  }
  try
    VisionPlan::new(
      [{ time: 1, height: 2, width: 2, }],
      max_workspace_bytes=1L,
      max_output_bytes=1L,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("budget accepted")
  }
}

///|
test "vision checkpoint groups include pre and post shuffle norms" {
  assert_eq(frontend_weights().regions[0].bytes, 1152L * 1536L * 2L)
  assert_eq(block_weights(26).regions.length(), 12)
  assert_eq(merger_weights(0).regions[0].bytes, 1152L * 2L)
  assert_eq(merger_weights(1).regions[0].bytes, 4608L * 2L)
  assert_eq(
    merger_weights(3).regions[0].name,
    "model.visual.deepstack_merger_list.2.norm.weight",
  )
  for
    group in [
      frontend_weights(),
      block_weights(0),
      merger_weights(0),
      merger_weights(1),
    ] {
    let last = group.regions[group.regions.length() - 1]
    assert_eq(last.offset + last.bytes, group.bytes)
  }
}
