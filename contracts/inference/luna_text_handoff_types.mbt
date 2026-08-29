///|
const LUNA_TEXT_HANDOFF_IDLE : Int = 0

///|
const LUNA_TEXT_HANDOFF_WRITING : Int = 1

///|
const LUNA_TEXT_HANDOFF_SEMANTIC : Int = 2

///|
const LUNA_TEXT_HANDOFF_READY : Int = 3

///|
const LUNA_TEXT_HANDOFF_LEASED : Int = 4

///|
const LUNA_TEXT_HANDOFF_CLAIMED : Int = 5

///|
pub struct LunaTextRequestHandoffStorage {
  priv limits : InferenceLimits
  priv model_identity : @spec.ModelIdentity
  priv cache_scope : FixedArray[Byte]
  priv cache_scope_length : Int
  priv prompt : FixedArray[Byte]
  priv semantics : LunaRequestSemanticStorage
  priv semantic_writes : Array[LunaRequestSemanticWrite]
  priv semantic_works : Array[LunaRequestSemanticWork]
  priv semantic_leases : Array[LunaRequestSemanticLease]
  priv mut epoch : UInt64
  priv mut phase : Int
  priv mut prompt_limit : Int
  priv mut prompt_length : Int
  priv mut prompt_cursor : Int
  priv mut request_id : RequestId
  priv mut effective : EffectiveLimits
  priv mut sampling : LunaValidatedSampling
  priv mut sampling_seed : SamplingSeed
  priv mut stream_preference : StreamPreference
  priv mut deadline : DeadlineBudget
}

///|
#valtype
pub struct LunaTextRequestHandoffWrite {
  priv storage : LunaTextRequestHandoffStorage
  priv epoch : UInt64
}

///|
#valtype
pub struct LunaTextRequestHandoffWork {
  priv storage : LunaTextRequestHandoffStorage
  priv epoch : UInt64
}

///|
#valtype
pub struct LunaTextRequestHandoffLease {
  priv storage : LunaTextRequestHandoffStorage
  priv epoch : UInt64
}

///|
#valtype
pub struct LunaTextRequestHandoffClaim {
  priv storage : LunaTextRequestHandoffStorage
  priv epoch : UInt64
}
