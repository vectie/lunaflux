///|
/// Independent counter domain used by latent initialization.
pub(all) enum JointLatentRandomStream {
  ConditioningStream
  VideoLatentStream
  AudioLatentStream
} derive(Debug, Eq)

///|
/// Projected row segment in the canonical packed transformer sequence.
pub(all) enum JointPackedRowSegment {
  ConditioningRows
  VideoRows
  AudioRows
} derive(Debug, Eq)

///|
/// Bounded correctness-reference failures. Payloads contain only bounded
/// indexes/counts or stable enum values, never user media or tensor contents.
pub(all) suberror JointDiffusionReferenceError {
  InvalidPlanContract
  InvalidRequestShape
  InvalidSchedule
  InvalidConditioningDrawLimit
  InvalidRandomCounter(JointLatentRandomStream)
  ShapeProductOverflow
  InvalidConditioningRows
  InvalidProjectedRowCount(JointPackedRowSegment)
  PackedElementLimitExceeded
  StageMismatch(
    expected~ : @joint_diffusion_plan.JointDiffusionStageKind,
    actual~ : @joint_diffusion_plan.JointDiffusionStageKind
  )
  RequiredStageCannotBeSkipped(@joint_diffusion_plan.JointDiffusionStageKind)
  StageMachineComplete
  StageMachineCancelled
  CompletedMachineCannotBeCancelled
} derive(Debug, Eq)

///|
/// Normalized shifted Rectified Flow sigma sequence and plan-scaled model
/// timesteps. The terminal sigma has no corresponding denoiser call.
pub struct ShiftedRectifiedFlowSchedule {
  priv latent : @joint_diffusion_plan.JointDiffusionLatentKind
  priv plan_digest : @spec.PlanDigest
  priv sigmas : ReadOnlyArray[Double]
  priv timesteps : ReadOnlyArray[Double]
} derive(Debug, Eq)

///|
/// Stateless counter-based normal generator bound to one plan and request.
pub struct CounterBasedJointLatentRng {
  priv plan_digest : @spec.PlanDigest
  priv key : UInt64
  priv conditioning_draw_limit : Int64
  priv video_draw_limit : Int64
  priv audio_draw_limit : Int64
} derive(Debug, Eq)

///|
/// Canonical projected-row layout: conditioning, video, then audio. All three
/// projected segments share the denoiser hidden width.
pub struct JointPackedRowLayout {
  priv plan_digest : @spec.PlanDigest
  priv hidden_width : Int
  priv conditioning_rows : Int
  priv video_rows : Int
  priv audio_rows : Int
  priv video_input_width : Int
  priv audio_input_width : Int
} derive(Debug, Eq)

///|
/// Bounded materialized reference of projected packed rows.
pub struct JointProjectedRows {
  priv layout : JointPackedRowLayout
  priv values : ReadOnlyArray[Double]
} derive(Debug, Eq)

///|
/// Split copies of the three projected row segments.
pub struct JointProjectedRowSplit {
  priv conditioning : ReadOnlyArray[Double]
  priv video : ReadOnlyArray[Double]
  priv audio : ReadOnlyArray[Double]
} derive(Debug, Eq)

///|
/// Immutable bounded stage cursor over the exact plan stage sequence.
pub struct JointDiffusionStageMachine {
  priv plan_digest : @spec.PlanDigest
  priv stages : ReadOnlyArray[@joint_diffusion_plan.JointDiffusionStage]
  priv next_index : Int
  priv cancelled : Bool
} derive(Debug, Eq)
