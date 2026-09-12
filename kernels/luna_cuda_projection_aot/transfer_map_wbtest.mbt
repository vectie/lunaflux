///|
test "CUDA transfer stride is specialized from the finite owner map" {
  for lanes in [32, 64, 128, 256, 512] {
    let source = projection_transfer_map_loop(512, lanes)
    assert_true(source.contains("#pragma unroll"))
    assert_true(source.contains("vector < 512"))
    assert_true(source.contains("vector += \{lanes}"))
    assert_false(source.contains("blockDim"))
  }
}
