///|
test "query square law changes only learned query kernel and its launch" {
  let make = law => {
    @quantized_query_kv_cuda_source.quantized_query_kv_source(
      @precision.QuantizedQueryKvPrecision::new(
        hidden_width=256,
        query_rank=128,
        head_count=2,
        head_width=128,
        maximum_rows=3,
        normalization_epsilon=1.0e-6,
        activation_amax_floor=0.0001,
        maximum_workspace_bytes=6144L,
        query_square_law=law,
      ),
      symbol_prefix="query_law",
    )
  }
  let baseline = make(VectorizedTreeFloat32(partitions=256, vector_width=1))
  let selected = make(VectorizedTreeFloat32(partitions=128, vector_width=4))
  assert_eq(selected.launch(1), (3, 1, 128))
  assert_eq(selected.launch(5), (3, 1, 256))
  let old = @utf8.decode(baseline.source_bytes())
    .split("extern \"C\" __global__ void ")
    .collect()
  let new = @utf8.decode(selected.source_bytes())
    .split("extern \"C\" __global__ void ")
    .collect()
  assert_eq(old.length(), 7)
  assert_eq(new.length(), 7)
  for i in 0..<7 {
    if i != 2 {
      assert_eq(new[i], old[i])
    }
  }
  assert_true(new[2].contains("float sums[4]"))
  try make(CompensatedFloat32) catch {
    @norm.PrefixRmsNormCudaSourceError::UnsupportedSquareReductionLaw => ()
    e => raise e
  } noraise {
    _ => fail("unimplemented query fold silently substituted")
  }
}
