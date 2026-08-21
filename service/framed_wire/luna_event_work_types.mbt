///|
const LUNA_FRAMED_EVENT_IDLE : Int = 0

///|
const LUNA_FRAMED_EVENT_CLEARING : Int = 1

///|
const LUNA_FRAMED_EVENT_SETUP : Int = 2

///|
const LUNA_FRAMED_EVENT_COPYING : Int = 3

///|
const LUNA_FRAMED_EVENT_CHECKSUM : Int = 4

///|
const LUNA_FRAMED_EVENT_FINALIZING : Int = 5

///|
const LUNA_FRAMED_EVENT_READY : Int = 6

///|
const LUNA_FRAMED_EVENT_VIEW_LEASED : Int = 7

///|
const LUNA_FRAMED_EVENT_FAILED : Int = 8

///|
const LUNA_FRAMED_EVENT_MAX_STEP_WORK_UNITS : Int = 65536

///|
#valtype
pub struct LunaFramedEventStepBudget {
  priv work_units : Int
}

///|
pub fn LunaFramedEventStepBudget::new(
  work_units : Int,
) -> LunaFramedEventStepBudget raise FramedWireError {
  if work_units <= 0 || work_units > LUNA_FRAMED_EVENT_MAX_STEP_WORK_UNITS {
    raise InvalidLimit(LunaFramedEventWorkUnits)
  }
  { work_units, }
}

///|
pub fn LunaFramedEventStepBudget::work_units(
  self : LunaFramedEventStepBudget,
) -> Int {
  self.work_units
}

///|
#valtype
pub(all) enum LunaFramedEventWorkState {
  LunaFramedEventPending
  LunaFramedEventReady
  LunaFramedEventFailed
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaFramedEventProgress {
  LunaFramedEventIdle
  LunaFramedEventAdvanced
} derive(Debug, Eq)

///|
pub struct LunaFramedEventWorkspace {
  priv limits : FramedWireLimits
  priv budget : LunaFramedEventStepBudget
  priv storage : FixedArray[Byte]
  priv semantic_views : Array[@luna_event.LunaEventView]
  priv mut epoch : UInt64
  priv mut phase : Int
  priv mut event_kind : Int
  priv mut length : Int
  priv mut payload_length : Int
  priv mut cursor : Int
  priv mut checksum : UInt
  priv mut failure_rule : Int
  priv mut last_work_units : Int
  priv mut total_work_units : UInt64
}

///|
#valtype
pub struct LunaFramedEventWork {
  priv workspace : LunaFramedEventWorkspace
  priv epoch : UInt64
}

///|
#valtype
pub struct LunaFramedEventView {
  priv workspace : LunaFramedEventWorkspace
  priv epoch : UInt64
}
