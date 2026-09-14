///|
test "fence fusion requires independent query-owned single-slot lifetimes" {
  for owned in [false, true] {
    for slots in [0, 1, 2, 3] {
      for
        readiness in [
          @luna_attention_tile_schedule.JoinedKeyValueReadiness,
          KeyScoreBeforeValueReadiness,
        ] {
        let fused = @luna_attention_tile_schedule.plan_attention_fold_fences(
            readiness,
            query_owned=owned,
            transfer_slots=slots,
          ) ==
          NextKeyPublicationAtValueRelease
        assert_eq(
          fused,
          owned && slots == 1 && readiness == KeyScoreBeforeValueReadiness,
        )
      }
    }
  }
}
