///|
test "CUDA renderer copies raw cells with unrolled constant index terms" {
  let plan = @dense_permutation_plan.Permutation::new(
    [2, 3],
    [1, 0],
    max_elements=6L,
  )
  let bytes = @dense_permutation_cuda_source.render(plan, block_size=128)
  let source = @utf8.decode(bytes)
  assert_true(source.contains("i >= 6ULL"))
  assert_true(source.contains("blockIdx.x * 128ULL"))
  assert_true(source.contains("((i / 2ULL) % 3ULL)"))
  assert_true(source.contains("(i % 2ULL) * 3ULL"))
  assert_true(source.contains("output[i] = input[src]"))
  assert_false(source.contains("float"))
  assert_false(source.contains("__syncthreads"))
  assert_eq(bytes, @dense_permutation_cuda_source.render(plan, block_size=128))
  assert_true(
    bytes !=
    @dense_permutation_cuda_source.render(plan.inverse(), block_size=128),
  )
}

///|
test "identity elides index arithmetic and launch geometry is bounded" {
  let plan = @dense_permutation_plan.Permutation::new(
    [2, 1, 3],
    [0, 2, 1],
    max_elements=6L,
  )
  let source = @utf8.decode(
    @dense_permutation_cuda_source.render(plan, block_size=256),
  )
  assert_true(source.contains("output[i] = input[i]"))
  assert_false(source.contains("src +="))
  for block_size in [0, -1, 1025] {
    try @dense_permutation_cuda_source.render(plan, block_size~) catch {
      @dense_permutation_cuda_source.SourceError::InvalidBlockSize => ()
    } noraise {
      _ => fail("invalid block size admitted")
    }
  }
}
