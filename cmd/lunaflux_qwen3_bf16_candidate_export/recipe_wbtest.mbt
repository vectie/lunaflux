///|
test "single attention recipe cannot masquerade as a partitioned launch" {
  let record = qwen_compiler_single_launch_record(46, 16, 256, 70032L)
  assert_eq(record, "\ngrid=46,16,1\nblock=256,1,1\nshared_memory_bytes=70032")
  assert_false(record.contains("partial_shared_memory_bytes="))
  assert_false(record.contains("merge_shared_memory_bytes="))
}

///|
test "partitioned attention recipe carries both entry-point launch resources" {
  let record = qwen_compiler_partitioned_launch_record(32, 16, 2, 256, 70032L)
  assert_eq(
    record,
    "\npartial_grid=32,16,2\nmerge_grid=32,16,1\nblock=256,1,1" +
    "\npartial_shared_memory_bytes=70032\nmerge_shared_memory_bytes=0",
  )
  assert_false(record.contains("\nshared_memory_bytes="))
}
