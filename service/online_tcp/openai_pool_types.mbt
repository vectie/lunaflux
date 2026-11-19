///|
const LUNA_OPENAI_POOL_FREE : Int32 = 0

///|
const LUNA_OPENAI_POOL_HTTP_RECEIVE : Int32 = 1

///|
const LUNA_OPENAI_POOL_HTTP_PROGRESS : Int32 = 2

///|
const LUNA_OPENAI_POOL_INBOUND_PROGRESS : Int32 = 3

///|
const LUNA_OPENAI_POOL_HANDOFF_FEED : Int32 = 4

///|
const LUNA_OPENAI_POOL_REQUEST_ACTIVE : Int32 = 5

///|
const LUNA_OPENAI_POOL_COMPAT_PROGRESS : Int32 = 6

///|
const LUNA_OPENAI_POOL_RESPONSE_PROGRESS : Int32 = 7

///|
const LUNA_OPENAI_POOL_OUTPUT_WRITE : Int32 = 8

///|
const LUNA_OPENAI_POOL_RETIRING : Int32 = 9

///|
const LUNA_OPENAI_POOL_CONTROL_WRITE : Int32 = 10

///|
#valtype
pub(all) enum LunaOnlineOpenAIConnectionPoolState {
  LunaOnlineOpenAIConnectionPoolListening
  LunaOnlineOpenAIConnectionPoolActive
  LunaOnlineOpenAIConnectionPoolDraining
  LunaOnlineOpenAIConnectionPoolClosed
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaOnlineOpenAIConnectionPoolReadiness {
  LunaOnlineOpenAIConnectionPoolNotReady
  LunaOnlineOpenAIConnectionPoolReady
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaOnlineOpenAIConnectionPoolProgress {
  LunaOnlineOpenAIConnectionAccepted
  LunaOnlineOpenAIConnectionPoolAdvanced
  LunaOnlineOpenAIConnectionPoolBackpressured
  LunaOnlineOpenAIConnectionRetired
  LunaOnlineOpenAIConnectionPoolDrainingProgress
  LunaOnlineOpenAIConnectionPoolClosedProgress
} derive(Debug, Eq)

///|
pub struct LunaOnlineOpenAIConnectionPoolLimits {
  priv connection_capacity : Int32
  priv output_buffer_bytes : Int32
  priv io_poll_timeout_millis : Int32
}

///|
priv struct LunaOnlineOpenAIConnectionSlot {
  receipt_workspace : @request_admission.LunaRequestReceiptWorkspace
  http_workspace : @http1.LunaHttp1Workspace
  response_workspace : @http1.LunaHttp1ResponseWorkspace
  inbound_workspace : @openai_compat.LunaOpenAIInboundWorkspace
  outbound_workspace : @openai_compat.LunaOpenAICompatWorkspace
  embedded_control : LunaOpenAIEmbeddedControl
  connections : Array[@socket.Tcp]
  receipts : Array[@request_admission.LunaRequestReceipt]
  http_works : Array[@http1.LunaHttp1Work]
  response_works : Array[@http1.LunaHttp1ResponseWork]
  response_views : Array[@http1.LunaHttp1ResponseView]
  inbound_works : Array[@openai_compat.LunaOpenAIInboundWork]
  inbound_views : Array[@openai_compat.LunaOpenAIInboundView]
  outbound_works : Array[@openai_compat.LunaOpenAICompatWork]
  outbound_views : Array[@openai_compat.LunaOpenAICompatView]
  semantic_events : Array[@online_session.LunaOnlineFramedSemanticEvent]
  rejections : Array[@online_session.LunaOnlineFramedRejectionCredit]
  routes : Array[@online_session.LunaOnlineFramedRequestRoute]
  input : FixedArray[Byte]
  output : LunaOnlineTcpOutputScratch
  output_writes : Array[LunaOnlineTcpOutputWrite]
  flights : Array[LunaOnlineTcpOutputFlight]
  outbound_copy_quanta : Int32
  mut phase : Int32
  mut generation : UInt64
  mut request_sequence : UInt64
  mut route_mode : Int32
  mut streaming_response : Bool
  mut tail_start : Int32
  mut tail_length : Int32
  mut response_kind : Int32
  mut response_copy_cursor : Int32
  mut compat_copy_cursor : Int32
  mut output_length : Int32
  mut output_cursor : Int32
  mut event_terminal : Bool
  mut terminal_observed : Bool
  mut sse_started : Bool
  mut cancel_requested : Bool
}

///|
pub struct LunaOnlineOpenAIConnectionPoolPreparation {
  priv service : @online_session.LunaOnlineFramedServicePreparation
  priv slots : FixedArray[LunaOnlineOpenAIConnectionSlot]
  priv limits : LunaOnlineOpenAIConnectionPoolLimits
  priv maximum_wait_millis : Int32
  priv auth_policy : @api_auth.LunaApiAuthPolicy
  priv production_ready : Bool
  priv mut consumed : Bool
}

///|
pub struct LunaOnlineOpenAIConnectionPool {
  priv tcp_limits : LunaOnlineTcpLimits
  priv pool_limits : LunaOnlineOpenAIConnectionPoolLimits
  priv bound_addr : @socket.Addr
  priv listeners : Array[@socket.TcpServer]
  priv services : Array[@online_session.LunaOnlineFramedService]
  priv streams : Array[@online_session.LunaOnlineFramedStream]
  priv slots : FixedArray[LunaOnlineOpenAIConnectionSlot]
  priv auth_policy : @api_auth.LunaApiAuthPolicy
  priv telemetry : LunaServerTelemetryBridge
  priv mut active_connections : Int32
  priv mut ingress_slot : Int32
  priv mut io_cursor : Int32
  priv mut accept_turn : Bool
  priv mut service_waiting : Bool
  priv mut next_request_sequence : UInt64
  priv mut drain_requested : Bool
  priv mut failure : (String, Error)?
  priv mut stream_disconnect_started : Bool
  priv mut service_drain_started : Bool
  priv mut closed : Bool
  priv production_ready : Bool
  priv mut cold_start_latency_millis : Int32
}
