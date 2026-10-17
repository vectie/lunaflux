///|
test "maximum publication bounds the second level" {
  let tree = @physical_tile_ir.MaximumReductionTree::new(leaves=256, group=32).unwrap()
  assert_eq(tree.geometry(), (256, 32, 8))
  assert_true(@physical_tile_ir.MaximumReductionTree::new(leaves=1024, group=16) is None)
  assert_true(@physical_tile_ir.MaximumReductionTree::new(leaves=256, group=31) is None)
  assert_true(@physical_tile_ir.MaximumReductionTree::new(leaves=255, group=32) is None)
}
