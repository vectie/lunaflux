///|
test "CUDA transfer stride is specialized from the finite owner map" {
  for lanes in [32, 64, 128, 256, 512] {
    let source = projection_physical_transfer_map_loop(
      @physical_tile_ir.VectorOwnership::new(vectors=512, owners=lanes).unwrap(),
    )
    assert_true(source.contains("#pragma unroll"))
    assert_true(source.contains("vector < 512"))
    assert_true(source.contains("vector += \{lanes}"))
    assert_false(source.contains("blockDim"))
  }
}
