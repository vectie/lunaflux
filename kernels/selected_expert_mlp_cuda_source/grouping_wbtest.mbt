///|
test "stable grouping publishes uniquely owned rows after the ordered prefix" {
  let small = grouping_kernel("test", 64, 8, 4096L)
  assert_true(small.contains("first[512],ordinal[512],offset[512]"))
  assert_true(small.contains("offset[first[job]]+rank/16"))
  assert_true(small.contains("if(rank%16==0) meta[1+group]"))
  assert_eq(small.split("__syncthreads()").count(), 3)
  // Larger capacities retain existing coverage rather than exceeding local
  // storage or silently shrinking the logical domain.
  let large = grouping_kernel("large", 1025, 1, 4096L)
  assert_true(large.contains("if(threadIdx.x!=0) return"))
}
