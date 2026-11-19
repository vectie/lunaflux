///|
pub(all) enum InheritedDrainRule {
  DescriptorUnavailable
  InvalidDescriptor
  InvalidState
  ChannelLost
} derive(Debug, Eq)

///|
pub(all) suberror InheritedDrainError {
  InheritedDrainFailed(InheritedDrainRule)
} derive(Debug, Eq)

///|
#valtype
pub(all) enum InheritedDrainProgress {
  InheritedDrainPending
  InheritedDrainRequested
  InheritedDrainResponseConfirmed
  InheritedDrainChannelLost
  InheritedDrainClosed
} derive(Debug, Eq)

///|
#valtype
pub(all) enum InheritedDrainResponse {
  InheritedDrainAccepted
  InheritedDrainAlreadyDraining
  InheritedDrainAlreadyClosed
} derive(Debug, Eq)

///|
pub struct InheritedDrainPreparation {
  priv native : @native.InheritedDrainHandle
  priv input : FixedArray[Byte]
  priv output : FixedArray[Byte]
  priv mut consumed : Bool
}

///|
pub struct InheritedDrainChannel {
  priv native : @native.InheritedDrainHandle
  priv input : FixedArray[Byte]
  priv output : FixedArray[Byte]
  priv mut input_cursor : Int32
  priv mut output_cursor : Int32
  priv mut phase : Int32
  priv mut request_latched : Bool
  priv mut response_started : Bool
}

///|
pub extend InheritedDrainRule with @debug.Debug::{to_repr}

///|
pub extend InheritedDrainRule with Eq::{not_equal, equal}

///|
pub extend InheritedDrainError with @debug.Debug::{to_repr}

///|
pub extend InheritedDrainError with Eq::{not_equal, equal}

///|
pub extend InheritedDrainProgress with @debug.Debug::{to_repr}

///|
pub extend InheritedDrainProgress with Eq::{not_equal, equal}

///|
pub extend InheritedDrainResponse with @debug.Debug::{to_repr}

///|
pub extend InheritedDrainResponse with Eq::{not_equal, equal}
