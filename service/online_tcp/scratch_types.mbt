///|
const LUNA_ONLINE_TCP_SCRATCH_IDLE : Int = 0

///|
const LUNA_ONLINE_TCP_SCRATCH_MUTATING : Int = 1

///|
const LUNA_ONLINE_TCP_SCRATCH_IN_FLIGHT : Int = 2

///|
const LUNA_ONLINE_TCP_MAX_SCRATCH_BYTES : Int = 16777216

///|
priv suberror LunaOnlineTcpScratchError {
  LunaOnlineTcpInvalidScratchCapacity
  LunaOnlineTcpScratchBusy
  LunaOnlineTcpScratchStale
  LunaOnlineTcpScratchRange
  LunaOnlineTcpScratchEpochExhausted
}

///|
priv struct LunaOnlineTcpOutputScratch {
  immutable : Bytes
  mutable : FixedArray[Byte]
  capacity : Int
  mut generation : UInt64
  mut phase : Int
  mut length : Int
}

///|
#valtype
priv struct LunaOnlineTcpOutputWrite {
  scratch : LunaOnlineTcpOutputScratch
  generation : UInt64
}

///|
#valtype
priv struct LunaOnlineTcpOutputFlight {
  scratch : LunaOnlineTcpOutputScratch
  generation : UInt64
  length : Int
}
