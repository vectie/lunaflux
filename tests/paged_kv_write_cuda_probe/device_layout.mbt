///|
const KV_ARENA_BYTES : Int64 = 267200L

///|
const KV_STEP : Int64 = 0L

///|
const KV_POSITIONS : Int64 = 32L

///|
const KV_ROW_OFFSETS : Int64 = 64L

///|
const KV_SEQUENCE_LENGTHS : Int64 = 96L

///|
const KV_PAGE_OFFSETS : Int64 = 128L

///|
const KV_PAGES : Int64 = 160L

///|
const KV_QKV_GUARD_BEFORE : Int64 = 256L

///|
const KV_QKV : Int64 = 320L

///|
const KV_QKV_GUARD_AFTER : Int64 = 4416L

///|
const KV_KEY_GUARD_BEFORE : Int64 = 4480L

///|
const KV_KEY : Int64 = 4544L

///|
const KV_KEY_GUARD_AFTER : Int64 = 135616L

///|
const KV_VALUE_GUARD_BEFORE : Int64 = 135680L

///|
const KV_VALUE : Int64 = 135744L

///|
const KV_VALUE_GUARD_AFTER : Int64 = 266816L

///|
const KV_CANARY : Int64 = 266944L

///|
const KV_CACHE_BYTES : Int64 = 131072L

///|
fn kv_region(
  arena : @device.Allocation,
  offset : Int64,
  bytes : Int64,
  alignment : Int64,
) -> @device.KernelArgument raise {
  @device.KernelArgument::device_region(
    arena,
    offset~,
    byte_count=bytes,
    alignment~,
  )
}

///|
fn kv_candidate_arguments(
  arena : @device.Allocation,
) -> @device.KernelArguments raise {
  @device.KernelArguments::new([
    kv_region(arena, KV_STEP, 20L, 4L),
    kv_region(arena, KV_POSITIONS, 32L, 4L),
    kv_region(arena, KV_ROW_OFFSETS, 20L, 4L),
    kv_region(arena, KV_SEQUENCE_LENGTHS, 16L, 4L),
    kv_region(arena, KV_PAGE_OFFSETS, 20L, 4L),
    kv_region(arena, KV_PAGES, 64L, 4L),
    kv_region(arena, KV_QKV, 4096L, 16L),
    kv_region(arena, KV_KEY, KV_CACHE_BYTES, 16L),
    kv_region(arena, KV_VALUE, KV_CACHE_BYTES, 16L),
    kv_region(arena, KV_CANARY, 4L, 4L),
  ])
}

///|
fn kv_oracle_arguments(
  arena : @device.Allocation,
) -> @device.KernelArguments raise {
  @device.KernelArguments::new([
    kv_region(arena, KV_STEP, 20L, 4L),
    kv_region(arena, KV_POSITIONS, 32L, 4L),
    kv_region(arena, KV_ROW_OFFSETS, 20L, 4L),
    kv_region(arena, KV_SEQUENCE_LENGTHS, 16L, 4L),
    kv_region(arena, KV_PAGE_OFFSETS, 20L, 4L),
    kv_region(arena, KV_PAGES, 64L, 4L),
    kv_region(arena, KV_QKV, 4096L, 16L),
    kv_region(arena, KV_KEY, KV_CACHE_BYTES, 16L),
    kv_region(arena, KV_VALUE, KV_CACHE_BYTES, 16L),
  ])
}
