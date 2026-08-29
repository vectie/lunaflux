///|
const LUNA_HTTP1_CONTROL_MIN_HEAD_BYTES : Int = 64

///|
const LUNA_HTTP1_CONTROL_MAX_HEAD_BYTES : Int = 4096

///|
const LUNA_HTTP1_CONTROL_IDLE : Int = 0

///|
const LUNA_HTTP1_CONTROL_RECEIVING : Int = 1

///|
const LUNA_HTTP1_CONTROL_READY : Int = 2

///|
const LUNA_HTTP1_CONTROL_VIEW_LEASED : Int = 3

///|
const LUNA_HTTP1_CONTROL_FAILED : Int = 4

///|
#valtype
pub(all) enum LunaHttp1ControlRoute {
  LunaHttp1Health
  LunaHttp1Readiness
  LunaHttp1Metrics
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaHttp1ControlRequestState {
  LunaHttp1ControlReceiving
  LunaHttp1ControlReady
  LunaHttp1ControlFailed
} derive(Debug, Eq)

///|
#valtype
pub struct LunaHttp1ControlLimits {
  priv max_head_bytes : Int
}

///|
pub struct LunaHttp1ControlWorkspace {
  priv storage : FixedArray[Byte]
  priv mut epoch : UInt64
  priv mut phase : Int
  priv mut length : Int
  priv mut terminator_state : Int
  priv mut route_code : Int
}

///|
#valtype
pub struct LunaHttp1ControlRequest {
  priv workspace : LunaHttp1ControlWorkspace
  priv epoch : UInt64
}

///|
#valtype
pub struct LunaHttp1ControlRequestView {
  priv workspace : LunaHttp1ControlWorkspace
  priv epoch : UInt64
}

///|
#valtype
pub(all) enum LunaHttp1ControlResponseKind {
  LunaHttp1ControlHealthy
  LunaHttp1ControlUnhealthy
  LunaHttp1ControlReadyResponse
  LunaHttp1ControlNotReadyResponse
  LunaHttp1ControlBadRequest
} derive(Debug, Eq)

///|
#valtype
pub struct LunaHttp1ControlResponse {
  priv kind : Int
}

///|
/// Bounded, payload-safe startup measurement document. The encoded response is
/// constructed once during startup publication and is reused by the control
/// server without request-path formatting or allocation.
pub struct LunaHttp1ControlMetricsResponse {
  priv status : Int
  priv encoded : Bytes
}
