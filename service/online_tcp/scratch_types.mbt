///|
const LUNA_ONLINE_TCP_SCRATCH_IDLE : Int32 = 0

///|
const LUNA_ONLINE_TCP_SCRATCH_MUTATING : Int32 = 1

///|
const LUNA_ONLINE_TCP_SCRATCH_IN_FLIGHT : Int32 = 2

///|
const LUNA_ONLINE_TCP_MAX_SCRATCH_BYTES : Int32 = 16777216

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
  capacity : Int32
  mut generation : UInt64
  mut phase : Int32
  mut length : Int32
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
  length : Int32
}
