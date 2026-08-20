///|
pub(all) enum LunaEventRule {
  LunaEventLifecycle
  LunaEventEpoch
  LunaEventKind
  LunaEventScalar
  LunaEventLength
  LunaEventPayload
  LunaEventDestination
} derive(Debug, Eq)

///|
pub(all) suberror LunaEventError {
  LunaEventInvalid(LunaEventRule)
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaEventPublishStatus {
  LunaEventPublishReady
  LunaEventPublishPayloadInvalid
  LunaEventPublishEpochExhausted
  LunaEventPublishCreditUnavailable
} derive(Debug, Eq)

///|
#valtype
priv enum LunaEventSlotKind {
  LunaNoEventSlot
  LunaAcceptedEventSlot
  LunaTokenEventSlot
  LunaUsageEventSlot
  LunaCompletedEventSlot
  LunaFailedEventSlot
} derive(Eq)

///|
pub struct LunaEventOwner {
  priv limits : @inference.InferenceLimits
  priv payload : FixedArray[Byte]
  priv mut model_identity : @spec.ModelIdentity?
  priv mut effective_limits : @inference.EffectiveLimits?
  priv mut token_id : Int
  priv mut position : Int
  priv mut input_tokens : Int
  priv mut cached_input_tokens : Int
  priv mut output_tokens : Int
  priv mut finish_reason : @inference.FinishReason
  priv mut retryable : Bool
  priv mut payload_length : Int
  priv mut epoch : UInt64
  priv mut kind : LunaEventSlotKind
  priv mut busy : Bool
}

///|
#valtype
pub(all) enum LunaEventKind {
  LunaAcceptedEventKind
  LunaTokenEventKind
  LunaUsageEventKind
  LunaCompletedEventKind
  LunaFailedEventKind
} derive(Debug, Eq)

///|
#valtype
pub struct LunaEventView {
  priv owner : LunaEventOwner
  priv epoch : UInt64
}

///|
#valtype
pub struct LunaAcceptedEventView {
  priv owner : LunaEventOwner
  priv epoch : UInt64
}

///|
#valtype
pub struct LunaTokenEventView {
  priv owner : LunaEventOwner
  priv epoch : UInt64
}

///|
#valtype
pub struct LunaUsageEventView {
  priv owner : LunaEventOwner
  priv epoch : UInt64
}

///|
#valtype
pub struct LunaCompletedEventView {
  priv owner : LunaEventOwner
  priv epoch : UInt64
}

///|
#valtype
pub struct LunaFailedEventView {
  priv owner : LunaEventOwner
  priv epoch : UInt64
}
