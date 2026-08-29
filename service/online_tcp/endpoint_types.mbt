///|
const LUNA_ONLINE_TCP_LISTENING : Int = 0

///|
const LUNA_ONLINE_TCP_ACCEPTING : Int = 1

///|
const LUNA_ONLINE_TCP_CONNECTED : Int = 2

///|
const LUNA_ONLINE_TCP_DRAINING : Int = 3

///|
const LUNA_ONLINE_TCP_CLOSE_REQUIRED : Int = 4

///|
const LUNA_ONLINE_TCP_CLOSED : Int = 5

///|
const LUNA_ONLINE_TCP_MAX_TIMEOUT_MILLIS : Int = 86400000

///|
const LUNA_ONLINE_TCP_MAX_TRANSITIONS : Int = 65536

///|
pub(all) enum LunaOnlineTcpRule {
  LunaOnlineTcpLimits
  LunaOnlineTcpPreparation
  LunaOnlineTcpBind
  LunaOnlineTcpAccept
  LunaOnlineTcpRead
  LunaOnlineTcpWrite
  LunaOnlineTcpTimeout
  LunaOnlineTcpCancelled
  LunaOnlineTcpCoordinator
  LunaOnlineTcpTelemetry
  LunaOnlineTcpLifecycle
  LunaOnlineTcpOwnership
} derive(Debug, Eq)

///|
pub(all) suberror LunaOnlineTcpError {
  LunaOnlineTcpFailed(LunaOnlineTcpRule)
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaOnlineTcpEndpointState {
  LunaOnlineTcpListening
  LunaOnlineTcpAccepting
  LunaOnlineTcpConnected
  LunaOnlineTcpDraining
  LunaOnlineTcpCloseRequired
  LunaOnlineTcpClosed
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaOnlineTcpProgress {
  LunaOnlineTcpConnectedProgress
  LunaOnlineTcpAdvanced
  LunaOnlineTcpBackpressured
  LunaOnlineTcpMaintenanceRequired
  LunaOnlineTcpRejected
  LunaOnlineTcpCleanupRequired
  LunaOnlineTcpClosedProgress
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaOnlineTcpRejectionRule {
  LunaOnlineTcpRejectedFrame
  LunaOnlineTcpRejectedIdentity
  LunaOnlineTcpRejectedDeadline
  LunaOnlineTcpRejectedInput
  LunaOnlineTcpRejectedTokenization
  LunaOnlineTcpRejectedCapacity
  LunaOnlineTcpRejectedService
} derive(Debug, Eq)

///|
pub struct LunaOnlineTcpLimits {
  priv read_chunk_bytes : Int
  priv write_chunk_bytes : Int
  priv accept_timeout_millis : Int
  priv input_idle_timeout_millis : Int
  priv write_timeout_millis : Int
  priv reactor_transition_budget : Int
}

///|
priv struct LunaOnlineTcpActivity {
  mut reactor_active : Bool
  mut maintenance_active : Bool
}

///|
priv struct LunaOnlineTcpMaintenanceHandoff {
  mut pending : Bool
}

///|
pub struct LunaOnlineTcpEndpoint {
  priv limits : LunaOnlineTcpLimits
  priv mut bound_addr : @socket.Addr
  priv servers : Array[@socket.TcpServer]
  priv connections : Array[@socket.Tcp]
  priv coordinators : Array[@online_session.LunaOnlineFramedCoordinator]
  priv input : FixedArray[Byte]
  priv output : LunaOnlineTcpOutputScratch
  priv offers : Array[@online_session.LunaOnlineFramedEventOffer]
  priv flights : Array[LunaOnlineTcpOutputFlight]
  priv activity : LunaOnlineTcpActivity
  priv mut phase : Int
  priv mut tail_start : Int
  priv mut tail_length : Int
  priv mut tail_backpressured : Bool
  priv maintenance_handoff : LunaOnlineTcpMaintenanceHandoff
  priv mut close_pending : Bool
  priv mut drain_requested : Bool
  priv mut disconnect_requested : Bool
  priv mut failure_rule : Int
  priv mut rejection_sequence : UInt64
  priv mut rejection_rule : Int
  priv mut rejection_latched : Bool
}
