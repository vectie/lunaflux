///|
const LUNA_OPENAI_CONNECTION_NONE : Int = 0

///|
const LUNA_OPENAI_HTTP_RECEIVE : Int = 1

///|
const LUNA_OPENAI_HTTP_PROGRESS : Int = 2

///|
const LUNA_OPENAI_INBOUND_PROGRESS : Int = 3

///|
const LUNA_OPENAI_HANDOFF_FEED : Int = 4

///|
const LUNA_OPENAI_STREAM_PROGRESS : Int = 5

///|
const LUNA_OPENAI_COMPAT_PROGRESS : Int = 6

///|
const LUNA_OPENAI_RESPONSE_PROGRESS : Int = 7

///|
const LUNA_OPENAI_RESPONSE_WRITE : Int = 8

///|
const LUNA_OPENAI_COMPAT_WRITE : Int = 9

///|
const LUNA_OPENAI_CONTROL_WRITE : Int = 10

///|
pub struct LunaOnlineOpenAIServerPreparation {
  priv service : @online_session.LunaOnlineFramedServicePreparation
  priv receipt : @request_admission.LunaRequestReceiptWorkspace
  priv http : @http1.LunaHttp1Workspace
  priv response : @http1.LunaHttp1ResponseWorkspace
  priv inbound : @openai_compat.LunaOpenAIInboundWorkspace
  priv outbound : @openai_compat.LunaOpenAICompatWorkspace
  priv embedded_control : LunaOpenAIEmbeddedControl
  priv production_ready : Bool
  priv auth_policy : @api_auth.LunaApiAuthPolicy
  priv mut consumed : Bool
  priv maximum_wait_millis : Int
  priv response_copy_quanta : Int
  priv outbound_copy_quanta : Int
}

///|
pub struct LunaOnlineOpenAIServer {
  priv limits : LunaOnlineTcpLimits
  priv bound_addr : @socket.Addr
  priv listeners : Array[@socket.TcpServer]
  priv connections : Array[@socket.Tcp]
  priv services : Array[@online_session.LunaOnlineFramedService]
  priv streams : Array[@online_session.LunaOnlineFramedStream]
  priv receipt_workspace : @request_admission.LunaRequestReceiptWorkspace
  priv receipts : Array[@request_admission.LunaRequestReceipt]
  priv http_workspace : @http1.LunaHttp1Workspace
  priv http_works : Array[@http1.LunaHttp1Work]
  priv response_workspace : @http1.LunaHttp1ResponseWorkspace
  priv response_works : Array[@http1.LunaHttp1ResponseWork]
  priv response_views : Array[@http1.LunaHttp1ResponseView]
  priv inbound_workspace : @openai_compat.LunaOpenAIInboundWorkspace
  priv inbound_works : Array[@openai_compat.LunaOpenAIInboundWork]
  priv inbound_views : Array[@openai_compat.LunaOpenAIInboundView]
  priv outbound_workspace : @openai_compat.LunaOpenAICompatWorkspace
  priv auth_policy : @api_auth.LunaApiAuthPolicy
  priv outbound_works : Array[@openai_compat.LunaOpenAICompatWork]
  priv outbound_views : Array[@openai_compat.LunaOpenAICompatView]
  priv embedded_control : LunaOpenAIEmbeddedControl
  priv semantic_events : Array[@online_session.LunaOnlineFramedSemanticEvent]
  priv rejections : Array[@online_session.LunaOnlineFramedRejectionCredit]
  priv observations : Array[@online_session.LunaOnlineFramedObservation]
  priv input : FixedArray[Byte]
  priv output : LunaOnlineTcpOutputScratch
  priv flights : Array[LunaOnlineTcpOutputFlight]
  priv telemetry : LunaServerTelemetryBridge
  priv activity : LunaOnlineTcpActivity
  priv response_copy_quanta : Int
  priv outbound_copy_quanta : Int
  priv mut phase : Int
  priv mut connection_phase : Int
  priv mut connection_epoch : UInt64
  priv mut request_sequence : UInt64
  priv mut route_mode : Int
  priv mut tail_start : Int
  priv mut tail_length : Int
  priv mut response_kind : Int
  priv mut output_source : Int
  priv mut output_cursor : Int
  priv mut output_length : Int
  priv mut event_terminal : Bool
  priv mut terminal_observed : Bool
  priv mut sse_started : Bool
  priv mut backpressure_episode_active : Bool
  priv mut connection_telemetry_active : Bool
  priv mut observation_recorded : Bool
  priv mut disconnect_pending : Bool
  priv mut disconnect_progress_pending : Bool
  priv mut drain_requested : Bool
  priv mut close_listener_pending : Bool
  priv mut service_drain_started : Bool
  priv mut drain_logged : Bool
  priv mut closed_logged : Bool
  priv mut failure_rule : Int
  priv production_ready : Bool
  priv mut cold_start_latency_millis : Int
}
