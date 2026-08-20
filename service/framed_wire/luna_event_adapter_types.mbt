///|
/// One transport-owned publication credit for a canonical event-v2 frame.
/// The adapter borrows semantic event evidence but never owns its retirement
/// or acknowledgement authority.
pub struct LunaFramedEventAdapter {
  priv limits : FramedWireLimits
  priv storage : FixedArray[Byte]
  priv mut busy : Bool
  priv mut length : Int
}
