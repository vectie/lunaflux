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
  priv mut input_cursor : Int
  priv mut output_cursor : Int
  priv mut phase : Int
  priv mut request_latched : Bool
  priv mut response_started : Bool
}
