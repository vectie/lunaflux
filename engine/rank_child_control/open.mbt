///|
/// Opens the inherited fixed-buffer child control channel. The caller must
/// still perform local artifact/device bootstrap admission before `begin_ready`.
pub fn open_rank_child_control(
  wire_limits : @rank_group_wire.RankGroupWireLimits,
  io_timeout_millis~ : Int,
  shutdown_timeout_millis~ : Int,
) -> RankChildControl raise RankChildControlError {
  let frame_bytes = wire_limits.max_frame_bytes()
  let process_limits = @process.ProcessLimits::new(
    max_frame_bytes=frame_bytes,
    io_timeout_millis~,
    shutdown_timeout_millis~,
  ) catch {
    _ => raise Invalid(Limit)
  }
  let channel = @process.open_inherited(process_limits)
  {
    channel,
    wire_limits,
    rx_destination: FixedArray::make(frame_bytes, b'\x00'),
    rx_staging: FixedArray::make(frame_bytes, b'\x00'),
    tx_storage: FixedArray::make(frame_bytes, b'\x00'),
    transcript: @rank_group_wire.RankGroupWireRankTranscript::new(),
    receive_active: false,
    receive_held: false,
    receive_held_eof: false,
    receive_held_length: 0,
    inbound_active: false,
    inbound_epoch: 0UL,
    inbound_kind: Configure,
    inbound_slot: None,
    inbound_sequence: 0UL,
    inbound_payload_length: 0,
    send_active: false,
    last_graph_sidecar_sequence: 0UL,
    failed_value: false,
    failure_rule_value: Lifecycle,
  }
}
