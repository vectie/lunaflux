///|
/// Shared-score initialization, before any asynchronous transfer is issued.
/// Query-bound validation borrows score storage. Its reads must finish before
/// the score fold or the first transport epoch can reuse that storage.
pub(all) enum AttentionScoreBootstrapEffect {
  InitializeScoreOutput
  PublishBootstrapWrites
  InitializeResidentScoreOutput
  StageScoreQueries
  InitializeScoreStatistics
  ValidateScoreQueryBounds
  ReadScoreQueryBounds
  RequireValidScoreQueries
  ReleaseScoreValidationReaders
} derive(Debug, Eq)

///|
pub(all) enum AttentionScoreBootstrapPoint {
  InitializeScoreAccumulators
  InitializeScoreOperands
  ValidateScoreOperands
} derive(Debug, Eq)

///|
pub fn plan_attention_score_bootstrap(
  point : AttentionScoreBootstrapPoint,
  resident_output~ : Bool,
) -> ReadOnlyArray[AttentionScoreBootstrapEffect] {
  match point {
    InitializeScoreAccumulators =>
      if resident_output {
        [
          InitializeScoreOutput,
          PublishBootstrapWrites,
          InitializeResidentScoreOutput,
          PublishBootstrapWrites,
        ]
      } else {
        [InitializeScoreOutput, PublishBootstrapWrites]
      }
    InitializeScoreOperands =>
      [StageScoreQueries, InitializeScoreStatistics, PublishBootstrapWrites]
    ValidateScoreOperands =>
      [
        ValidateScoreQueryBounds,
        PublishBootstrapWrites,
        ReadScoreQueryBounds,
        RequireValidScoreQueries,
        ReleaseScoreValidationReaders,
      ]
  }
}
