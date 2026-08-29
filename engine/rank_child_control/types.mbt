///|
pub(all) enum RankChildControlRule {
  Limit
  Lifecycle
  Binding
  Epoch
  Payload
  ReceiveBusy
  SendBusy
  Transport
  Wire
  LocalAdmission
} derive(Debug, Eq)

///|
pub(all) suberror RankChildControlError {
  Invalid(RankChildControlRule)
  Failed(RankChildControlRule)
} derive(Debug, Eq)

///|
#valtype
pub struct RankChildReceiveProgress {
  priv kind : Int
}

///|
#valtype
pub struct RankChildSendProgress {
  priv kind : Int
}

///|
pub struct RankChildControl {
  priv channel : @process.InheritedChannel
  priv wire_limits : @rank_group_wire.RankGroupWireLimits
  priv rx_destination : FixedArray[Byte]
  priv rx_staging : FixedArray[Byte]
  priv tx_storage : FixedArray[Byte]
  priv transcript : @rank_group_wire.RankGroupWireRankTranscript
  priv mut receive_active : Bool
  priv mut receive_held : Bool
  priv mut receive_held_eof : Bool
  priv mut receive_held_length : Int
  priv mut inbound_active : Bool
  priv mut inbound_epoch : UInt64
  priv mut inbound_kind : @rank_group_wire.RankGroupWireKind
  priv mut inbound_slot : @rank_group_wire.RankGroupWireSlot?
  priv mut inbound_sequence : UInt64
  priv mut inbound_payload_length : Int
  priv mut send_active : Bool
  priv mut last_graph_sidecar_sequence : UInt64
  priv mut failed_value : Bool
  priv mut failure_rule_value : RankChildControlRule
}

///|
#valtype
pub struct RankChildInbound(RankChildControl, UInt64)

///|
pub fn RankChildReceiveProgress::is_pending(
  self : RankChildReceiveProgress,
) -> Bool {
  self.kind == 0
}

///|
pub fn RankChildReceiveProgress::is_held(
  self : RankChildReceiveProgress,
) -> Bool {
  self.kind == 1
}

///|
pub fn RankChildReceiveProgress::has_inbound(
  self : RankChildReceiveProgress,
) -> Bool {
  self.kind == 2
}

///|
pub fn RankChildReceiveProgress::is_clean_eof(
  self : RankChildReceiveProgress,
) -> Bool {
  self.kind == 3
}

///|
pub fn RankChildSendProgress::is_pending(self : RankChildSendProgress) -> Bool {
  self.kind == 0
}

///|
pub fn RankChildSendProgress::is_complete(self : RankChildSendProgress) -> Bool {
  self.kind == 1
}

///|
pub fn RankChildControl::is_failed(self : RankChildControl) -> Bool {
  self.failed_value
}

///|
pub fn RankChildControl::failure_rule(
  self : RankChildControl,
) -> RankChildControlRule? {
  if self.failed_value {
    Some(self.failure_rule_value)
  } else {
    None
  }
}

///|
pub fn RankChildControl::lifecycle(
  self : RankChildControl,
) -> @rank_group_wire.RankGroupWireLifecycle {
  self.transcript.lifecycle()
}
