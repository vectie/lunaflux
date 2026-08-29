///|
const PROBE_ARENA_BYTES : Int64 = 98304L

///|
const PROBE_STEP_COUNTS : Int64 = 0L

///|
const PROBE_QUERY_POSITIONS : Int64 = 32L

///|
const PROBE_QUERY_ROW_OFFSETS : Int64 = 64L

///|
const PROBE_SEQUENCE_LENGTHS : Int64 = 80L

///|
const PROBE_PAGE_TABLE_OFFSETS : Int64 = 96L

///|
const PROBE_PAGE_TABLE_INDICES : Int64 = 112L

///|
const PROBE_INPUT : Int64 = 128L

///|
const PROBE_Q_WEIGHT : Int64 = 2176L

///|
const PROBE_K_WEIGHT : Int64 = 34944L

///|
const PROBE_V_WEIGHT : Int64 = 51328L

///|
const PROBE_ROTATED_QKV : Int64 = 67712L

///|
const PROBE_KEY_CACHE : Int64 = 73728L

///|
const PROBE_VALUE_CACHE : Int64 = 81920L

///|
const PROBE_QKV_CANARY : Int64 = 90112L

///|
const PROBE_RESIDUAL : Int64 = 128L

///|
const PROBE_BRANCH : Int64 = 2176L

///|
const PROBE_NORM_WEIGHT : Int64 = 4224L

///|
const PROBE_RESIDUAL_OUTPUT : Int64 = 4608L

///|
const PROBE_NORM_OUTPUT : Int64 = 6656L

///|
const PROBE_RESIDUAL_CANARY : Int64 = 8704L

///|
const PROBE_STANDALONE_PROJECTED_QKV : Int64 = 90128L

///|
const PROBE_ATTENTION_OUTPUT : Int64 = 94240L

///|
fn probe_upload(
  arena : @device.Allocation,
  offset : Int64,
  bytes : Bytes,
) -> Unit raise {
  arena.copy_from_host(bytes, 0L, offset, bytes.length().to_int64())
}

///|
fn probe_read_bf16(
  arena : @device.Allocation,
  offset : Int64,
  elements : Int,
) -> Array[Int] raise {
  probe_bf16_bits(arena.copy_to_host(offset, (elements * 2).to_int64()))
}
