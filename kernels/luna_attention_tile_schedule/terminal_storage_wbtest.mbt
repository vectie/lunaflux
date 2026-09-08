///|
test "resident output borrows input storage only in prologue and terminal phases" {
  for candidate_id in [312, 313, 316] {
    let program = schedule_test_program(candidate_id~)
    let layout = plan_attention_matrix_storage(program)
    assert_eq(layout.offset(AccumulatorStorage), layout.offset(QueryStorage))
    assert_eq(
      layout.offset(AccumulatorLayoutProbeStorage),
      layout.offset(QueryStorage),
    )
    for span in layout.spans() {
      match span.role() {
        AccumulatorStorage => assert_eq(span.lifetime(), (4, 5))
        AccumulatorLayoutProbeStorage => assert_eq(span.lifetime(), (-1, 0))
        QueryStorage => assert_eq(span.lifetime(), (0, 4))
        _ => ()
      }
    }
    assert_eq(
      layout.offset(TerminalMaximumStorage),
      layout.offset(ScoreStorage),
    )
    assert_eq(
      layout.peak_bytes(),
      match candidate_id {
        312 => 28672L
        313 => 49152L
        _ => 47248L
      },
    )
  }
}
