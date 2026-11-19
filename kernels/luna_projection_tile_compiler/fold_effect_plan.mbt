///|
/// Priming is distinct from iteration: no current operand may be consumed
/// before initial producers are ready and their writes are published.
pub enum ProjectionFoldBootstrapEffect {
  IssueInitialOperands
  AwaitInitialOperands
  PublishInitialOperands
} derive(Debug, Eq)

///|
/// Transfer completion is an explicit effect, independent of instruction set.
/// Overlap issues a future immutable operand before consuming the current one;
/// serial transfer issues it afterward. Both publish before the next consume.
pub(all) enum ProjectionTransferExecution {
  SerialTransfer
  CompletedPrefetch
  OverlappedTransfer
} derive(Debug, Eq)

///|
/// Ordered effects of one ring iteration. Publication is workgroup-wide;
/// consumption retains the original ordered arithmetic body.
pub(all) enum ProjectionFoldEffect {
  IssueFutureOperand
  ConsumeCurrentOperand
  AwaitOperandTransfers
  PublishOperandSlot
} derive(Debug, Eq)

///|
/// Finite, immutable control plan for a ring of operand materializations.
/// Counts describe transfer groups, not CUDA async instructions or subgroups.
pub struct ProjectionFoldEffectPlan {
  priv stages : Int32
  priv width : Int32
  priv extent : Int32
  priv execution : ProjectionTransferExecution
} derive(Debug, Eq)

///|
pub fn ProjectionFoldPipeline::effect_plan(
  self : ProjectionFoldPipeline,
  execution : ProjectionTransferExecution,
) -> ProjectionFoldEffectPlan {
  {
    stages: self.stage_count(),
    width: self.transfer_width(),
    extent: self.transfer_count() * self.transfer_width(),
    execution,
  }
}

///|
/// (ring slots, transfer width, reduction extent).
pub fn ProjectionFoldEffectPlan::geometry(
  self : ProjectionFoldEffectPlan,
) -> (Int32, Int32, Int32) {
  (self.stages, self.width, self.extent)
}

///|
pub fn ProjectionFoldEffectPlan::prime_count(
  self : ProjectionFoldEffectPlan,
) -> Int32 {
  self.stages - 1
}

///|
pub fn ProjectionFoldEffectPlan::future_distance(
  self : ProjectionFoldEffectPlan,
) -> Int32 {
  self.prime_count() * self.width
}

///|
pub fn ProjectionFoldEffectPlan::overlaps(
  self : ProjectionFoldEffectPlan,
) -> Bool {
  self.execution != SerialTransfer
}

///|
/// Issuing a synchronous prefetch already completes its writes. Publication
/// still requires a workgroup barrier, but no asynchronous completion event.
pub fn ProjectionFoldEffectPlan::has_pending_transfers(
  self : ProjectionFoldEffectPlan,
) -> Bool {
  self.execution == OverlappedTransfer
}

///|
/// At most this many younger issued groups may remain pending when the
/// current operand is consumed. At the tail, drain all groups instead.
pub fn ProjectionFoldEffectPlan::pending_group_limit(
  self : ProjectionFoldEffectPlan,
) -> Int32 {
  if self.has_pending_transfers() {
    self.stages - 2
  } else {
    0
  }
}

///|
/// The compiler owns effect ordering. A backend maps each action to its own
/// transfer/wait/barrier primitives without deciding where to move the fold.
pub fn ProjectionFoldEffectPlan::iteration(
  self : ProjectionFoldEffectPlan,
) -> ReadOnlyArray[ProjectionFoldEffect] {
  match self.execution {
    SerialTransfer =>
      [ConsumeCurrentOperand, IssueFutureOperand, PublishOperandSlot]
    CompletedPrefetch =>
      [IssueFutureOperand, ConsumeCurrentOperand, PublishOperandSlot]
    OverlappedTransfer =>
      [
        IssueFutureOperand,
        ConsumeCurrentOperand,
        AwaitOperandTransfers,
        PublishOperandSlot,
      ]
  }
}

///|
pub fn ProjectionFoldEffectPlan::bootstrap(
  self : ProjectionFoldEffectPlan,
) -> ReadOnlyArray[ProjectionFoldBootstrapEffect] {
  match self.execution {
    SerialTransfer | CompletedPrefetch =>
      [IssueInitialOperands, PublishInitialOperands]
    OverlappedTransfer =>
      [IssueInitialOperands, AwaitInitialOperands, PublishInitialOperands]
  }
}

///|
pub extend ProjectionFoldBootstrapEffect with @debug.Debug::{to_repr}

///|
pub extend ProjectionFoldBootstrapEffect with Eq::{not_equal, equal}

///|
pub extend ProjectionTransferExecution with @debug.Debug::{to_repr}

///|
pub extend ProjectionTransferExecution with Eq::{not_equal, equal}

///|
pub extend ProjectionFoldEffect with @debug.Debug::{to_repr}

///|
pub extend ProjectionFoldEffect with Eq::{not_equal, equal}

///|
pub extend ProjectionFoldEffectPlan with @debug.Debug::{to_repr}

///|
pub extend ProjectionFoldEffectPlan with Eq::{not_equal, equal}
