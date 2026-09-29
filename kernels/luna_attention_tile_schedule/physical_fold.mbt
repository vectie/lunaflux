///|
/// Lower the selected query ownership into an explicit physical register-state
/// program before device realization. Other ownership families have distinct
/// storage/exchange contracts and cannot silently acquire this dialect.
fn AttentionTileSchedule::physical_online_fold(
  self : AttentionTileSchedule,
) -> @attention_physical_ir.OnlineFold? {
  guard self.fragment_ownership() == QueryOwnedFragments else { return None }
  let keys = for combinator in self.loops {
    if combinator.axis == AttentionKeyValueAxis {
      break combinator.tile
    }
  } nobreak {
    0
  }
  @attention_physical_ir.OnlineFold::new(
    queries=self.loops[0].tile,
    keys~,
    dimensions=self.loops[self.loops.length() - 1].extent,
  )
}
