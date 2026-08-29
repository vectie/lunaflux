///|
/// Immutable startup plan for identically indexed rank-local KV arenas.
///
/// It owns no page table, allocator, device allocation, or backend handle.
pub struct TensorParallelKvPlan {
  priv identity : @spec.ModelIdentity
  priv generation_value : UInt64
  priv world_size : Int
  priv global_kv_head_count : Int
  priv local_kv_head_count : Int
  priv local_layout : @device_layout.DeviceKvLayout
  priv full_arena_bytes : Int64
  priv rank_kv_reservations : ReadOnlyArray[Int64]
  priv rank_bindings : ReadOnlyArray[TensorParallelKvRankBinding]
} derive(Debug, Eq)

///|
/// Opaque digest of one canonical rank-local KV bootstrap contract.
pub struct TensorParallelKvContractDigest {
  priv sha256 : String
} derive(Debug, Eq)

///|
/// Output-only proof binding one admitted rank to the shared logical-page
/// geometry and its exact local arena. It has no public constructor.
pub struct TensorParallelKvRankBinding {
  priv identity : @spec.ModelIdentity
  priv generation_value : UInt64
  priv rank : Int
  priv world_size : Int
  priv process_ordinal : Int
  priv target : @catalog.DeviceTarget
  priv global_kv_head_count : Int
  priv local_kv_head_count : Int
  priv layer_count : Int
  priv head_dimension : Int
  priv layout_version : Int
  priv tokens_per_page : Int
  priv total_page_count : Int
  priv page_logical_bytes : Int64
  priv page_stride_bytes : Int64
  priv component_stride_bytes : Int64
  priv layer_stride_bytes : Int64
  priv arena_bytes : Int64
  priv kv_reservation_bytes : Int64
  priv digest : TensorParallelKvContractDigest
} derive(Debug, Eq)

///|
/// One authority-free rank-local byte region for an allocator-issued logical
/// page identity. The exact page generation is retained unchanged.
#valtype
pub struct TensorParallelKvPageRegion {
  priv rank : Int
  priv page_index : Int
  priv page_generation : Int
  priv offset_bytes : Int64
  priv logical_bytes : Int64
  priv reserved_bytes : Int64
} derive(Debug, Eq)

///|
struct RankDeviceFacts {
  identity : @spec.ModelIdentity
  rank : Int
  world_size : Int
  device_ordinal : Int
  target : @catalog.DeviceTarget
} derive(Debug, Eq)

///|
struct RankRuntimeFacts {
  rank : Int
  world_size : Int
  process_ordinal : Int
  target : @catalog.DeviceTarget
  kv_reserved_bytes : Int64
} derive(Debug, Eq)
