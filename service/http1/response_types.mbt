///|
const LUNA_HTTP1_RESPONSE_CAPACITY : Int = 167

///|
const LUNA_HTTP1_RESPONSE_IDLE : Int = 0

///|
const LUNA_HTTP1_RESPONSE_BUILDING : Int = 1

///|
const LUNA_HTTP1_RESPONSE_FINALIZING : Int = 2

///|
const LUNA_HTTP1_RESPONSE_READY : Int = 3

///|
const LUNA_HTTP1_RESPONSE_VIEW_LEASED : Int = 4

///|
#valtype
pub(all) enum LunaHttp1ErrorResponseKind {
  LunaHttp1AuthenticationError
  LunaHttp1ValidationError
  LunaHttp1CapacityError
  LunaHttp1ServiceError
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaHttp1ResponseWorkState {
  LunaHttp1ResponseBuilding
  LunaHttp1ResponseReady
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaHttp1ResponseProgress {
  LunaHttp1ResponseIdle
  LunaHttp1ResponseAdvanced
} derive(Debug, Eq)

///|
#valtype
pub struct LunaHttp1ResponseStepBudget {
  priv work_units : Int
}

///|
pub struct LunaHttp1ResponseWorkspace {
  priv budget : LunaHttp1ResponseStepBudget
  priv storage : FixedArray[Byte]
  priv mut epoch : UInt64
  priv mut phase : Int
  priv mut response_kind : Int
  priv mut status_code : Int
  priv mut response_length : Int
  priv mut cursor : Int
  priv mut last_work_units : Int
  priv mut total_work_units : UInt64
}

///|
#valtype
pub struct LunaHttp1ResponseWork {
  priv workspace : LunaHttp1ResponseWorkspace
  priv epoch : UInt64
}

///|
#valtype
pub struct LunaHttp1ResponseView {
  priv workspace : LunaHttp1ResponseWorkspace
  priv epoch : UInt64
}
