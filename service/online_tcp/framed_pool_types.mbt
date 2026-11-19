///|
const LUNA_FRAMED_POOL_FREE : Int32 = 0

///|
const LUNA_FRAMED_POOL_RECEIVING : Int32 = 1

///|
const LUNA_FRAMED_POOL_VALIDATING : Int32 = 2

///|
const LUNA_FRAMED_POOL_FEEDING : Int32 = 3

///|
const LUNA_FRAMED_POOL_ACTIVE : Int32 = 4

///|
const LUNA_FRAMED_POOL_ENCODING : Int32 = 5

///|
const LUNA_FRAMED_POOL_COPYING : Int32 = 6

///|
const LUNA_FRAMED_POOL_WRITING : Int32 = 7

///|
const LUNA_FRAMED_POOL_RETIRING : Int32 = 8

///|
#valtype
pub(all) enum LunaOnlineFramedConnectionPoolState {
  LunaOnlineFramedConnectionPoolListening
  LunaOnlineFramedConnectionPoolActive
  LunaOnlineFramedConnectionPoolDraining
  LunaOnlineFramedConnectionPoolFailed
  LunaOnlineFramedConnectionPoolClosed
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaOnlineFramedConnectionPoolReadiness {
  LunaOnlineFramedConnectionPoolNotReady
  LunaOnlineFramedConnectionPoolReady
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaOnlineFramedConnectionPoolProgress {
  LunaOnlineFramedConnectionAccepted
  LunaOnlineFramedConnectionPoolAdvanced
  LunaOnlineFramedConnectionPoolBackpressured
  LunaOnlineFramedConnectionRetired
  LunaOnlineFramedConnectionPoolDrainingProgress
  LunaOnlineFramedConnectionPoolFailedProgress
  LunaOnlineFramedConnectionPoolClosedProgress
} derive(Debug, Eq)

///|
pub struct LunaOnlineFramedConnectionPoolLimits {
  priv connection_capacity : Int32
  priv io_poll_timeout_millis : Int32
  priv request_work_units : Int32
  priv event_work_units : Int32
}

///|
priv struct LunaOnlineFramedConnectionSlot {
  receipt_workspace : @request_admission.LunaRequestReceiptWorkspace
  request_workspace : @framed_wire.LunaFramedRequestWorkspace
  event_workspace : @framed_wire.LunaFramedEventWorkspace
  connections : Array[@socket.Tcp]
  receipts : Array[@request_admission.LunaRequestReceipt]
  request_works : Array[@framed_wire.LunaFramedRequestWork]
  request_views : Array[@framed_wire.LunaFramedRequestView]
  event_works : Array[@framed_wire.LunaFramedEventWork]
  event_views : Array[@framed_wire.LunaFramedEventView]
  semantic_events : Array[@online_session.LunaOnlineFramedSemanticEvent]
  routes : Array[@online_session.LunaOnlineFramedRequestRoute]
  input : FixedArray[Byte]
  output : LunaOnlineTcpOutputScratch
  output_writes : Array[LunaOnlineTcpOutputWrite]
  flights : Array[LunaOnlineTcpOutputFlight]
  mut phase : Int32
  mut generation : UInt64
  mut input_cursor : Int32
  mut input_tail_start : Int32
  mut input_tail_length : Int32
  mut feed_cursor : Int32
  mut event_copy_cursor : Int32
  mut output_length : Int32
  mut output_cursor : Int32
  mut event_terminal : Bool
  mut terminal_delivered : Bool
  mut cancel_requested : Bool
}

///|
pub struct LunaOnlineFramedConnectionPoolPreparation {
  priv service : @online_session.LunaOnlineFramedServicePreparation
  priv slots : FixedArray[LunaOnlineFramedConnectionSlot]
  priv limits : LunaOnlineFramedConnectionPoolLimits
  priv maximum_wait_millis : Int32
  priv mut consumed : Bool
}

///|
pub struct LunaOnlineFramedConnectionPool {
  priv tcp_limits : LunaOnlineTcpLimits
  priv pool_limits : LunaOnlineFramedConnectionPoolLimits
  priv bound_addr : @socket.Addr
  priv listeners : Array[@socket.TcpServer]
  priv services : Array[@online_session.LunaOnlineFramedService]
  priv streams : Array[@online_session.LunaOnlineFramedStream]
  priv slots : FixedArray[LunaOnlineFramedConnectionSlot]
  priv telemetry : LunaServerTelemetryBridge
  priv mut active_connections : Int32
  priv mut ingress_slot : Int32
  priv mut io_cursor : Int32
  priv mut accept_turn : Bool
  priv mut service_waiting : Bool
  priv mut service_probe_cooldown : Int32
  priv mut drain_requested : Bool
  priv mut failed : Bool
  priv mut stream_disconnect_started : Bool
  priv mut service_drain_started : Bool
  priv mut closed : Bool
  priv mut network_accepts : UInt64
  priv mut network_disconnects : UInt64
  priv mut delivered_events : UInt64
}
