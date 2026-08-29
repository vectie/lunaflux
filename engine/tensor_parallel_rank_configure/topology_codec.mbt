///|
const SELECTED_TOPOLOGY_MAGIC : UInt = 0x544c464c

///|
const SELECTED_TOPOLOGY_VERSION : UInt = 1

///|
const SELECTED_TOPOLOGY_HEADER_BYTES : Int = 32

///|
const SELECTED_TOPOLOGY_RANK_HEADER_BYTES : Int = 32

///|
const SELECTED_TOPOLOGY_MAX_NAME_BYTES : Int = 1024

///|
const SELECTED_TOPOLOGY_MAX_BYTES : Int = 16928

///|
fn selected_topology_length(
  topology : @device_topology.LocalTopologyDeclaration,
) -> Int raise TensorParallelRankConfigureError {
  let mut length = SELECTED_TOPOLOGY_HEADER_BYTES
  for rank in 0..<topology.rank_count() {
    guard topology.rank(rank) is Some(device) else { raise Invalid(Topology) }
    let name_length = @utf8.encode(device.name()).length()
    if name_length <= 0 || name_length > SELECTED_TOPOLOGY_MAX_NAME_BYTES {
      raise Invalid(Topology)
    }
    length += SELECTED_TOPOLOGY_RANK_HEADER_BYTES + name_length
  }
  if length > SELECTED_TOPOLOGY_MAX_BYTES {
    raise Invalid(Limit)
  }
  length
}

///|
fn selected_topology_checksum(
  bytes : FixedArray[Byte],
  offset : Int,
  length : Int,
) -> UInt {
  configure_checksum(bytes, offset, length, offset + 12)
}

///|
fn write_selected_topology(
  destination : FixedArray[Byte],
  offset : Int,
  topology : @device_topology.LocalTopologyDeclaration,
) -> Int raise TensorParallelRankConfigureError {
  let length = selected_topology_length(topology)
  configure_write_u32(destination, offset, SELECTED_TOPOLOGY_MAGIC)
  configure_write_u32(destination, offset + 4, SELECTED_TOPOLOGY_VERSION)
  configure_write_i32(destination, offset + 8, length)
  configure_write_i32(destination, offset + 16, topology.rank_count())
  configure_write_i32(destination, offset + 20, topology.link_count())
  configure_write_u64(destination, offset + 24, 0UL)
  let mut cursor = offset + SELECTED_TOPOLOGY_HEADER_BYTES
  for rank in 0..<topology.rank_count() {
    let device = topology.rank(rank).unwrap()
    let name = @utf8.encode(device.name())
    let target = device.target()
    configure_write_i32(destination, cursor, device.ordinal())
    configure_write_i32(destination, cursor + 4, name.length())
    configure_write_u64(
      destination,
      cursor + 8,
      device.total_memory_bytes().reinterpret_as_uint64(),
    )
    configure_write_i32(destination, cursor + 16, target.compute_major())
    configure_write_i32(destination, cursor + 20, target.compute_minor())
    let flags : UInt = (if target.supports_bf16() { 1 } else { 0 }) |
      (if target.supports_cublas_lt() { 2 } else { 0 })
    configure_write_u32(destination, cursor + 24, flags)
    configure_write_u32(destination, cursor + 28, 0)
    for index in 0..<name.length() {
      destination[cursor + SELECTED_TOPOLOGY_RANK_HEADER_BYTES + index] = name[index]
    }
    cursor += SELECTED_TOPOLOGY_RANK_HEADER_BYTES + name.length()
  }
  configure_write_u32(
    destination,
    offset + 12,
    selected_topology_checksum(destination, offset, length),
  )
  length
}

///|
fn decode_selected_topology(
  source : FixedArray[Byte],
  offset : Int,
  length : Int,
) -> @device_topology.LocalTopologyDeclaration raise TensorParallelRankConfigureError {
  if length < SELECTED_TOPOLOGY_HEADER_BYTES ||
    length > SELECTED_TOPOLOGY_MAX_BYTES ||
    offset < 0 ||
    offset > source.length() - length ||
    configure_read_i32(source, offset + 8) != length {
    raise Invalid(Length)
  }
  if configure_read_u32(source, offset) != SELECTED_TOPOLOGY_MAGIC {
    raise Invalid(Magic)
  }
  if configure_read_u32(source, offset + 4) != SELECTED_TOPOLOGY_VERSION {
    raise Invalid(Version)
  }
  if configure_read_u64(source, offset + 24) != 0UL {
    raise Invalid(Reserved)
  }
  if configure_read_u32(source, offset + 12) !=
    selected_topology_checksum(source, offset, length) {
    raise Invalid(Checksum)
  }
  let rank_count = configure_read_i32(source, offset + 16)
  if rank_count < 2 ||
    rank_count > 16 ||
    configure_read_i32(source, offset + 20) != rank_count * (rank_count - 1) / 2 {
    raise Invalid(Topology)
  }
  let ranks : Array[@device_topology.LocalTopologyRank] = []
  let end = offset + length
  let mut cursor = offset + SELECTED_TOPOLOGY_HEADER_BYTES
  for _rank in 0..<rank_count {
    if cursor > end - SELECTED_TOPOLOGY_RANK_HEADER_BYTES {
      raise Invalid(Length)
    }
    let name_length = configure_read_i32(source, cursor + 4)
    if name_length <= 0 ||
      name_length > SELECTED_TOPOLOGY_MAX_NAME_BYTES ||
      cursor + SELECTED_TOPOLOGY_RANK_HEADER_BYTES > end - name_length ||
      (configure_read_u32(source, cursor + 24) & 0xfffffffc) != 0 ||
      configure_read_u32(source, cursor + 28) != 0 {
      raise Invalid(Topology)
    }
    let name = @utf8.decode(
      Bytes::makei(name_length, index => {
        source[cursor + SELECTED_TOPOLOGY_RANK_HEADER_BYTES + index]
      }),
    ) catch {
      _ => raise Invalid(Topology)
    }
    let flags = configure_read_u32(source, cursor + 24)
    let target = @catalog.DeviceTarget::new(
      compute_major=configure_read_i32(source, cursor + 16),
      compute_minor=configure_read_i32(source, cursor + 20),
      supports_bf16=(flags & 1) != 0,
      supports_cublas_lt=(flags & 2) != 0,
    ) catch {
      _ => raise Invalid(Topology)
    }
    ranks.push(
      @device_topology.LocalTopologyRank::new(
        ordinal=configure_read_i32(source, cursor),
        name~,
        total_memory_bytes=configure_read_u64(source, cursor + 8).reinterpret_as_int64(),
        target~,
      ) catch {
        _ => raise Invalid(Topology)
      },
    )
    cursor += SELECTED_TOPOLOGY_RANK_HEADER_BYTES + name_length
  }
  if cursor != end {
    raise Invalid(Length)
  }
  let links : Array[@device_topology.LocalTopologyLink] = []
  for lower in 0..<rank_count {
    for upper in (lower + 1)..<rank_count {
      links.push(
        @device_topology.LocalTopologyLink::direct(
          lower_rank=lower,
          upper_rank=upper,
        ) catch {
          _ => raise Invalid(Topology)
        },
      )
    }
  }
  @device_topology.LocalTopologyDeclaration::new(ranks, links) catch {
    _ => raise Invalid(Topology)
  }
}
