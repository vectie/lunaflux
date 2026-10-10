///|
priv enum Phase {
  Ready
  Preparing
  Executing
  Restoring
  Committing
  Committed
  Releasing
  Failed
  Closed
}

///|
/// Model-neutral whole-step greedy verification. The coordinator saves both
/// ranks, joins activation and execution, then applies the pure prefix law.
/// Rejection restores physical state before replaying only accepted inputs.
/// It owns ingress, remote and activation effects. Snapshot lifetime remains
/// explicit: close_local_state must drain model work and resolve/release the
/// borrowed transaction before rank allocations are closed.
pub struct TwoRankGreedyVerification {
  priv ingress : @rank.SerialDecoderRank
  priv remote : @control.DecoderStageClient
  priv local_state : @control.StageStateTransaction
  priv close_local_state : () -> Unit raise
  priv begin_activation : (UInt64, Int, UInt64) -> Unit raise
  priv poll_activation : async (UInt64) -> Bool
  priv close_activation : () -> Unit raise
  priv original : @wire.PlanFrameBuffer
  priv replay : @wire.PlanFrameBuffer
  priv storage : FixedArray[Byte]
  priv completions : @wire.CompletionFrameBuffer
  priv inputs : FixedArray[Int]
  priv outputs : FixedArray[Int]
  priv stops : ReadOnlyArray[Int]
  priv vocabulary : Int
  priv timeout : UInt64
  priv mut phase : Phase
  priv mut busy : Bool
  priv mut epoch : UInt64
  priv mut tick : UInt64
  priv mut deadline : UInt64
  priv mut live : Int
  priv mut remaining : Int
  priv mut local_ready : Bool
  priv mut remote_ready : Bool
  priv mut activation_done : Bool
  priv mut save_begun : Bool
  priv mut save_done : Bool
  priv mut restore_done : Bool
  priv mut replaying : Bool
  priv mut failed : Bool
  priv mut decision : @greedy.VerifiedGreedyPrefix?
}

///|
pub fn TwoRankGreedyVerification::new(
  ingress : @rank.SerialDecoderRank,
  remote : @control.DecoderStageClient,
  limits : @protocol.WorkerProtocolLimits,
  inference : @inference.InferenceLimits,
  local_state~ : @control.StageStateTransaction,
  close_local_state~ : () -> Unit raise,
  begin_activation~ : (UInt64, Int, UInt64) -> Unit raise,
  poll_activation~ : async (UInt64) -> Bool,
  close_activation~ : () -> Unit raise,
  stop_tokens~ : ArrayView[Int],
  timeout_millis? : UInt64 = 300000UL,
) -> Self raise {
  let (rows, _, vocabulary) = ingress.ports().geometry()
  guard !ingress.outputs_token() &&
    remote.outputs_token() &&
    remote.output_capacity() == rows &&
    rows <= limits.max_plan_tokens() &&
    stop_tokens.length() <= inference.max_stop_token_ids() &&
    timeout_millis > 0UL &&
    timeout_millis <= 300000UL else {
    raise @control.StageControlError::InvalidConfiguration
  }
  for token in stop_tokens {
    guard token >= 0 && token < vocabulary else {
      raise @control.StageControlError::InvalidConfiguration
    }
  }
  let original = @wire.PlanFrameBuffer::new(limits, inference)
  {
    ingress,
    remote,
    local_state,
    close_local_state,
    begin_activation,
    poll_activation,
    close_activation,
    original,
    replay: @wire.PlanFrameBuffer::new(limits, inference),
    storage: FixedArray::make(original.capacity(), b'\x00'),
    completions: @wire.CompletionFrameBuffer::new(limits, inference),
    inputs: FixedArray::make(rows, 0),
    outputs: FixedArray::make(rows, 0),
    stops: ReadOnlyArray::from_array(stop_tokens.to_owned()),
    vocabulary,
    timeout: timeout_millis,
    phase: Ready,
    busy: false,
    epoch: 0UL,
    tick: 0UL,
    deadline: 0UL,
    live: 0,
    remaining: 0,
    local_ready: false,
    remote_ready: false,
    activation_done: false,
    save_begun: false,
    save_done: false,
    restore_done: false,
    replaying: false,
    failed: false,
    decision: None,
  }
}
