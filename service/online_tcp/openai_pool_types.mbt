///|
const LUNA_OPENAI_POOL_FREE : Int = 0

///|
const LUNA_OPENAI_POOL_HTTP_RECEIVE : Int = 1

///|
const LUNA_OPENAI_POOL_HTTP_PROGRESS : Int = 2

///|
const LUNA_OPENAI_POOL_INBOUND_PROGRESS : Int = 3

///|
const LUNA_OPENAI_POOL_HANDOFF_FEED : Int = 4

///|
const LUNA_OPENAI_POOL_REQUEST_ACTIVE : Int = 5

///|
const LUNA_OPENAI_POOL_COMPAT_PROGRESS : Int = 6

///|
const LUNA_OPENAI_POOL_RESPONSE_PROGRESS : Int = 7

///|
const LUNA_OPENAI_POOL_OUTPUT_WRITE : Int = 8

///|
const LUNA_OPENAI_POOL_RETIRING : Int = 9

///|
const LUNA_OPENAI_POOL_CONTROL_WRITE : Int = 10

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
  priv connection_capacity : Int
  priv output_buffer_bytes : Int
  priv io_poll_timeout_millis : Int
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
  outbound_copy_quanta : Int
  mut phase : Int
  mut generation : UInt64
  mut request_sequence : UInt64
  mut route_mode : Int
  mut tail_start : Int
  mut tail_length : Int
  mut response_kind : Int
  mut response_copy_cursor : Int
  mut compat_copy_cursor : Int
  mut output_length : Int
  mut output_cursor : Int
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
  priv maximum_wait_millis : Int
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
  priv mut active_connections : Int
  priv mut ingress_slot : Int
  priv mut io_cursor : Int
  priv mut accept_turn : Bool
  priv mut next_request_sequence : UInt64
  priv mut drain_requested : Bool
  priv mut stream_disconnect_started : Bool
  priv mut service_drain_started : Bool
  priv mut closed : Bool
  priv production_ready : Bool
  priv mut cold_start_latency_millis : Int
}
