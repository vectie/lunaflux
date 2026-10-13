///|
test "chunks cover the exact causal prefix once and sample only the last row" {
  for count in [1, 15, 16, 17, 22, 36, 63] {
    for capacity in [1, 2, 8, 16, 64] {
      let p = @prefill_physical_ir.ContiguousPrefillPlan::new(
        input_tokens=count,
        row_capacity=capacity,
        context_capacity=64,
      )
      let mut covered = 0
      let mut outputs = 0
      for index in 0..<p.chunk_count() {
        let (begin, end) = p.range(index)
        assert_eq(begin, covered)
        assert_true(end > begin && end - begin <= capacity)
        covered = end
        if p.produces_token(index) {
          outputs += 1
          assert_eq(end, count)
        }
      }
      assert_eq(covered, count)
      assert_eq(outputs, 1)
    }
  }
}
