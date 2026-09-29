///|
test "span cost rejects insufficient measurements and invalid estimates" {
  try
    @luna_fusion_strategy.FusionSpanCost::measured(
      median_nanoseconds=1L,
      samples=2,
    )
  catch {
    @luna_fusion_strategy.InvalidCost => ()
  } noraise {
    _ => fail("insufficient samples")
  }
  for terms in [(9223372036854775807L, 1L), (-1L, 1L), (0L, 0L)] {
    let (compute_nanoseconds, traffic_nanoseconds) = terms
    try
      @luna_fusion_strategy.FusionSpanCost::estimated(
        compute_nanoseconds~,
        traffic_nanoseconds~,
        synchronization_nanoseconds=0L,
        launch_nanoseconds=0L,
      )
    catch {
      @luna_fusion_strategy.InvalidCost => ()
    } noraise {
      _ => fail("invalid estimate")
    }
  }
}
