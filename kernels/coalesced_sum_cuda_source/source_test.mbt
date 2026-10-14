///|
test "coalesced collective copies bits without a floating point expression" {
  let p = @parallel.CoalescedSumPlan::new(
    maximum_rows=64,
    left_width=32768,
    right_width=4096,
  )
  let code = @coalesced_sum_cuda_source.source(p, symbol="batch")
  assert_true(code.contains("unsigned *left"))
  assert_false(code.contains("__fadd"))
  if @env.get_env_var("LUNA_EXPORT_REGIONS") == Some("1") {
    println("LUNA_SOURCE_BEGIN")
    println(code)
    println("LUNA_SOURCE_END")
  }
}
