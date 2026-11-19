///|
const LUNA_FRAMED_EVENT_IDLE : Int32 = 0

///|
const LUNA_FRAMED_EVENT_CLEARING : Int32 = 1

///|
const LUNA_FRAMED_EVENT_SETUP : Int32 = 2

///|
const LUNA_FRAMED_EVENT_COPYING : Int32 = 3

///|
const LUNA_FRAMED_EVENT_FINALIZING : Int32 = 5

///|
const LUNA_FRAMED_EVENT_READY : Int32 = 6

///|
const LUNA_FRAMED_EVENT_VIEW_LEASED : Int32 = 7

///|
const LUNA_FRAMED_EVENT_FAILED : Int32 = 8

///|
const LUNA_FRAMED_EVENT_MAX_STEP_WORK_UNITS : Int32 = 65536

///|
#valtype
pub struct LunaFramedEventStepBudget {
  priv work_units : Int32
}

///|
pub fn LunaFramedEventStepBudget::new(
  work_units : Int32,
) -> LunaFramedEventStepBudget raise FramedWireError {
  if work_units <= 0 || work_units > LUNA_FRAMED_EVENT_MAX_STEP_WORK_UNITS {
    raise InvalidLimit(LunaFramedEventWorkUnits)
  }
  { work_units, }
}

///|
pub fn LunaFramedEventStepBudget::work_units(
  self : LunaFramedEventStepBudget,
) -> Int32 {
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
  priv mut phase : Int32
  priv mut event_kind : Int32
  priv mut length : Int32
  priv mut payload_length : Int32
  priv mut cursor : Int32
  priv mut failure_rule : Int32
  priv mut last_work_units : Int32
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
