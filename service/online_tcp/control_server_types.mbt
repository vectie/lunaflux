///|
const LUNA_CONTROL_SERVER_LISTENING : Int = 0

///|
const LUNA_CONTROL_SERVER_ACCEPTING : Int = 1

///|
const LUNA_CONTROL_SERVER_RECEIVING : Int = 2

///|
const LUNA_CONTROL_SERVER_WRITING : Int = 3

///|
const LUNA_CONTROL_SERVER_DRAINING : Int = 4

///|
const LUNA_CONTROL_SERVER_CLOSED : Int = 5

///|
const LUNA_CONTROL_SERVER_FAILED : Int = 6

///|
priv enum LunaOnlineControlSource {
  LunaNativeControlSource(LunaOnlineTcpServer)
  LunaNativePoolControlSource(LunaOnlineFramedConnectionPool)
  LunaOpenAIControlSource(LunaOnlineOpenAIServer)
  LunaOpenAIPoolControlSource(LunaOnlineOpenAIConnectionPool)
}

///|
pub struct LunaOnlineHttpControlPreparation {
  priv source : LunaOnlineControlSource
  priv workspace : @http1.LunaHttp1ControlWorkspace
  priv production_ready : Bool
  priv mut consumed : Bool
}

///|
#valtype
pub(all) enum LunaOnlineHttpControlState {
  LunaOnlineHttpControlListening
  LunaOnlineHttpControlConnected
  LunaOnlineHttpControlDraining
  LunaOnlineHttpControlClosed
  LunaOnlineHttpControlFailed
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaOnlineHttpControlReadiness {
  LunaOnlineHttpControlNotReady
  LunaOnlineHttpControlReady
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaOnlineHttpControlProgress {
  LunaOnlineHttpControlAcceptTimedOut
  LunaOnlineHttpControlConnectedProgress
  LunaOnlineHttpControlAdvanced
  LunaOnlineHttpControlConnectionRetired
  LunaOnlineHttpControlDrainingProgress
  LunaOnlineHttpControlClosedProgress
  LunaOnlineHttpControlFailedProgress
} derive(Debug, Eq)

///|
pub struct LunaOnlineHttpControlServer {
  priv source : LunaOnlineControlSource
  priv production_ready : Bool
  priv limits : LunaOnlineTcpLimits
  priv bound_addr : @socket.Addr
  priv listeners : Array[@socket.TcpServer]
  priv connections : Array[@socket.Tcp]
  priv workspace : @http1.LunaHttp1ControlWorkspace
  priv requests : Array[@http1.LunaHttp1ControlRequest]
  priv input : FixedArray[Byte]
  priv responses : FixedArray[Bytes]
  priv metrics_output : LunaOnlineTcpOutputScratch
  priv metrics_flights : Array[LunaOnlineTcpOutputFlight]
  priv mut output : Bytes
  priv mut output_cursor : Int
  priv mut output_length : Int
  priv mut phase : Int
  priv mut drain_requested : Bool
  priv mut failed : Bool
  priv mut startup_measurement_published : Bool
  priv mut cold_start_latency_millis : Int
}
