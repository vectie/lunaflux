///|
const LUNA_ONLINE_TCP_SERVER_LISTENING : Int = 0

///|
const LUNA_ONLINE_TCP_SERVER_ACCEPTING : Int = 1

///|
const LUNA_ONLINE_TCP_SERVER_CONNECTED : Int = 2

///|
const LUNA_ONLINE_TCP_SERVER_RETIRING : Int = 3

///|
const LUNA_ONLINE_TCP_SERVER_DRAINING : Int = 4

///|
const LUNA_ONLINE_TCP_SERVER_CLOSED : Int = 5

///|
#valtype
pub(all) enum LunaOnlineTcpServerState {
  LunaOnlineTcpServerListening
  LunaOnlineTcpServerAccepting
  LunaOnlineTcpServerConnected
  LunaOnlineTcpServerRetiring
  LunaOnlineTcpServerDraining
  LunaOnlineTcpServerClosed
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaOnlineTcpServerProgress {
  LunaOnlineTcpServerListeningProgress
  LunaOnlineTcpServerAcceptTimedOut
  LunaOnlineTcpServerConnectedProgress
  LunaOnlineTcpServerAdvanced
  LunaOnlineTcpServerBackpressured
  LunaOnlineTcpServerObservationRecorded
  LunaOnlineTcpServerConnectionRetired
  LunaOnlineTcpServerDrainingProgress
  LunaOnlineTcpServerClosedProgress
} derive(Debug, Eq)

///|
/// Persistent one-listener, one-active-stream native framed server owner.
pub struct LunaOnlineTcpServer {
  priv limits : LunaOnlineTcpLimits
  priv mut bound_addr : @socket.Addr
  priv listeners : Array[@socket.TcpServer]
  priv connections : Array[@socket.Tcp]
  priv services : Array[@online_session.LunaOnlineFramedService]
  priv streams : Array[@online_session.LunaOnlineFramedStream]
  priv input : FixedArray[Byte]
  priv output : LunaOnlineTcpOutputScratch
  priv offers : Array[@online_session.LunaOnlineFramedEventOffer]
  priv flights : Array[LunaOnlineTcpOutputFlight]
  priv observations : Array[@online_session.LunaOnlineFramedObservation]
  priv telemetry : LunaServerTelemetryBridge
  priv activity : LunaOnlineTcpActivity
  priv pipeline_mode : Bool
  priv mut phase : Int
  priv mut tail_start : Int
  priv mut tail_length : Int
  priv mut tail_backpressured : Bool
  priv mut request_receipt_complete : Bool
  priv mut pipeline_receipt_observed : Bool
  priv mut pipeline_receipts_pending : Int
  priv mut backpressure_episode_active : Bool
  priv mut terminal_observed : Bool
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
}
