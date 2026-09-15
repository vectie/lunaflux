///|
test "terminal materialization publishes resident state before arena reuse" {
  for resident_output in [false, true] {
    for resident_statistics in [false, true] {
      let plan = @luna_attention_tile_schedule.AttentionScoreTerminalPlan::new(
        resident_output~,
        resident_statistics~,
      )
      let output = plan.actions(MaterializeScoreOutput)
      let statistics = plan.actions(MaterializeScoreStatistics)
      assert_eq(
        output.iter().to_array(),
        if resident_output {
          [StoreResidentOutput, PublishTerminalWrites]
        } else {
          []
        },
      )
      assert_eq(
        statistics.iter().to_array(),
        if resident_statistics {
          [StoreResidentStatistics, PublishTerminalWrites]
        } else {
          []
        },
      )
      assert_eq(plan.actions(ReleaseScoreArena).iter().to_array(), [
        ReleaseTerminalReaders,
      ])
    }
  }
}
