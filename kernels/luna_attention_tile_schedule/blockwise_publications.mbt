///|
/// Remove subgroup publications only when the enclosing effect schedule
/// already provides their visibility edge. K retirement follows all score
/// writes; V acquisition precedes the first probability reader. The intervening
/// softmax denominator operations are register-local and do not read another
/// worker's probability. No arithmetic, transfer or workgroup edge is removed.
pub fn AttentionGroupedEffectPlan::blockwise_steps(
  self : AttentionGroupedEffectPlan,
  fold : @attention_physical_ir.BlockwiseFold,
) -> ReadOnlyArray[@attention_physical_ir.BlockwiseFoldStep] {
  let scores_published = self
    .actions(ReleaseKeyReaders)
    .contains(PublishWorkgroup)
  let probabilities_published = self
    .actions(AcquireValueOperands)
    .contains(PublishWorkgroup)
  ReadOnlyArray::from_array(
    [
      for step in fold.steps() if !(step == PublishTileScores &&
      scores_published) &&
      !(step == PublishTileProbabilities && probabilities_published) => step
    ],
  )
}
