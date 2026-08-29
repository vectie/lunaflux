///|
fn append_u64(output : Array[Byte], value : UInt64) -> Unit {
  for byte in value.to_le_bytes() {
    output.push(byte)
  }
}

///|
fn append_i64(output : Array[Byte], value : Int64) -> Unit {
  append_u64(output, value.reinterpret_as_uint64())
}

///|
fn append_int(output : Array[Byte], value : Int) -> Unit {
  append_i64(output, Int64::from_int(value))
}

///|
fn append_bool(output : Array[Byte], value : Bool) -> Unit {
  output.push(if value { 1 } else { 0 })
}

///|
fn append_text(output : Array[Byte], value : StringView) -> Unit {
  let encoded = @utf8.encode(value)
  append_int(output, encoded.length())
  for byte in encoded {
    output.push(byte)
  }
}

///|
fn build_rank_binding(
  identity : @spec.ModelIdentity,
  generation_value : UInt64,
  rank : Int,
  world_size : Int,
  process_ordinal : Int,
  target : @catalog.DeviceTarget,
  global_kv_head_count : Int,
  local_layout : @device_layout.DeviceKvLayout,
  kv_reservation_bytes : Int64,
) -> TensorParallelKvRankBinding {
  let geometry = local_layout.geometry()
  let canonical : Array[Byte] = []
  for byte in b"lunaflux.tensor-parallel-kv-rank.v1\x00" {
    canonical.push(byte)
  }
  append_text(canonical, identity.content().as_hex())
  append_text(canonical, identity.plan().as_hex())
  append_u64(canonical, generation_value)
  append_int(canonical, rank)
  append_int(canonical, world_size)
  append_int(canonical, process_ordinal)
  append_int(canonical, target.compute_major())
  append_int(canonical, target.compute_minor())
  append_bool(canonical, target.supports_bf16())
  append_bool(canonical, target.supports_cublas_lt())
  append_int(canonical, global_kv_head_count)
  append_int(canonical, geometry.kv_head_count())
  append_int(canonical, geometry.layer_count())
  append_int(canonical, geometry.head_dimension())
  append_int(canonical, local_layout.stable_layout_version())
  append_int(canonical, local_layout.tokens_per_page())
  append_int(canonical, local_layout.total_page_count())
  append_i64(canonical, local_layout.page_logical_bytes())
  append_i64(canonical, local_layout.page_stride_bytes())
  append_i64(canonical, local_layout.component_stride_bytes())
  append_i64(canonical, local_layout.layer_stride_bytes())
  append_i64(canonical, local_layout.arena_bytes())
  append_i64(canonical, kv_reservation_bytes)
  let digest : TensorParallelKvContractDigest = {
    sha256: @crypto.bytes_to_hex_string(
      @crypto.sha256(Bytes::from_array(canonical)),
    ),
  }
  {
    identity,
    generation_value,
    rank,
    world_size,
    process_ordinal,
    target,
    global_kv_head_count,
    local_kv_head_count: geometry.kv_head_count(),
    layer_count: geometry.layer_count(),
    head_dimension: geometry.head_dimension(),
    layout_version: local_layout.stable_layout_version(),
    tokens_per_page: local_layout.tokens_per_page(),
    total_page_count: local_layout.total_page_count(),
    page_logical_bytes: local_layout.page_logical_bytes(),
    page_stride_bytes: local_layout.page_stride_bytes(),
    component_stride_bytes: local_layout.component_stride_bytes(),
    layer_stride_bytes: local_layout.layer_stride_bytes(),
    arena_bytes: local_layout.arena_bytes(),
    kv_reservation_bytes,
    digest,
  }
}
