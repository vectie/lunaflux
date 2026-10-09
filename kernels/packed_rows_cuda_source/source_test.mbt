///|
test "packing source uses unsigned cells mapping and explicit padding" {
  let plan = @packed_rows_plan.Plan::new(
    width=8,
    total_rows=5,
    conditioning=[0],
    video=[3, 4],
    audio=[1],
    max_elements=40L,
  )
  let source = @utf8.decode(
    @packed_rows_cuda_source.render(plan, block_size=128),
  )
  assert_true(source.contains("i >= 40ULL"))
  assert_true(source.contains("blockIdx.x * 128ULL"))
  assert_true(source.contains("tag == 0U) { output[i] = 0; return; }"))
  assert_true(source.contains("output[i] = input[source]"))
  assert_true(!source.contains("__syncthreads"))
  assert_true(!source.contains("float"))
  for block_size in [0, 1025] {
    try @packed_rows_cuda_source.render(plan, block_size~) catch {
      @packed_rows_cuda_source.SourceError::InvalidBlockSize => ()
    } noraise {
      _ => fail("invalid block admitted")
    }
  }
}
