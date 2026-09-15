///|
pub(all) enum AttentionScoreTerminalPoint {
  MaterializeScoreOutput
  MaterializeScoreStatistics
  ReleaseScoreArena
} derive(Debug, Eq)

///|
pub(all) enum AttentionScoreTerminalEffect {
  StoreResidentOutput
  StoreResidentStatistics
  PublishTerminalWrites
  ReleaseTerminalReaders
} derive(Debug, Eq)

///|
/// Cross-owner terminal reads require resident state to be materialized and
/// published. Arena reuse must wait for the terminal output readers.
pub struct AttentionScoreTerminalPlan {
  priv resident_output : Bool
  priv resident_statistics : Bool
} derive(Debug, Eq)

///|
pub fn AttentionScoreTerminalPlan::new(
  resident_output~ : Bool,
  resident_statistics~ : Bool,
) -> AttentionScoreTerminalPlan {
  { resident_output, resident_statistics, }
}

///|
pub fn AttentionScoreTerminalPlan::actions(
  self : AttentionScoreTerminalPlan,
  point : AttentionScoreTerminalPoint,
) -> ReadOnlyArray[AttentionScoreTerminalEffect] {
  match point {
    MaterializeScoreOutput =>
      if self.resident_output {
        [StoreResidentOutput, PublishTerminalWrites]
      } else {
        []
      }
    MaterializeScoreStatistics =>
      if self.resident_statistics {
        [StoreResidentStatistics, PublishTerminalWrites]
      } else {
        []
      }
    ReleaseScoreArena => [ReleaseTerminalReaders]
  }
}
