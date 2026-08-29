///|
priv struct KvProbeObservation {
  token_count : Int
  key_mutations : Int64
  value_mutations : Int64
  observed_canary : UInt
  launches : Int
  device_uuid : String
  device_pci : String
  device_name_sha256 : String
  device_total_memory_bytes : Int64
  cuda_driver_version : Int
}

///|
fn kv_upload(
  arena : @device.Allocation,
  offset : Int64,
  bytes : Bytes,
) -> Unit raise {
  arena.copy_from_host(bytes, 0L, offset, bytes.length().to_int64())
}

///|
fn kv_fill_bytes(count : Int, value : Byte) -> Bytes {
  Bytes::from_array(FixedArray::make(count, value)[:])
}

///|
fn kv_reset_case(
  arena : @device.Allocation,
  case_ : KvWriteCase,
) -> Array[Int] raise {
  kv_upload(arena, KV_STEP, kv_i32_bytes(case_.step_counts))
  kv_upload(arena, KV_POSITIONS, kv_i32_bytes(case_.positions))
  kv_upload(arena, KV_ROW_OFFSETS, kv_i32_bytes(case_.row_offsets))
  kv_upload(arena, KV_SEQUENCE_LENGTHS, kv_i32_bytes(case_.sequence_lengths))
  kv_upload(arena, KV_PAGE_OFFSETS, kv_i32_bytes(case_.page_offsets))
  kv_upload(arena, KV_PAGES, kv_i32_bytes(case_.pages))
  let qkv = kv_case_qkv_bits(case_)
  kv_upload(arena, KV_QKV_GUARD_BEFORE, kv_fill_bytes(64, b'\xa5'))
  kv_upload(arena, KV_QKV, kv_fill_bytes(4096, b'\xcc'))
  kv_upload(arena, KV_QKV, kv_bf16_bytes(qkv[:]))
  kv_upload(arena, KV_QKV_GUARD_AFTER, kv_fill_bytes(64, b'\xa5'))
  kv_reset_outputs(arena)
  qkv
}

///|
fn kv_reset_outputs(arena : @device.Allocation) -> Unit raise {
  let poison = kv_bf16_bytes(Array::make(65536, 0x7fc1)[:])
  kv_upload(arena, KV_KEY_GUARD_BEFORE, kv_fill_bytes(64, b'\xa5'))
  kv_upload(arena, KV_KEY, poison)
  kv_upload(arena, KV_KEY_GUARD_AFTER, kv_fill_bytes(64, b'\xa5'))
  kv_upload(arena, KV_VALUE_GUARD_BEFORE, kv_fill_bytes(64, b'\xa5'))
  kv_upload(arena, KV_VALUE, poison)
  kv_upload(arena, KV_VALUE_GUARD_AFTER, kv_fill_bytes(64, b'\xa5'))
  kv_upload(arena, KV_CANARY, b"\x00\x00\x00\x00")
}

///|
fn kv_guards_unchanged(arena : @device.Allocation) -> Bool raise {
  let expected_guard = kv_fill_bytes(64, b'\xa5')
  for
    offset in [
      KV_QKV_GUARD_BEFORE,
      KV_QKV_GUARD_AFTER,
      KV_KEY_GUARD_BEFORE,
      KV_KEY_GUARD_AFTER,
      KV_VALUE_GUARD_BEFORE,
      KV_VALUE_GUARD_AFTER,
    ] {
    if arena.copy_to_host(offset, 64L) != expected_guard {
      return false
    }
  }
  true
}

///|
fn kv_scalar_expected(
  case_ : KvWriteCase,
  qkv : ArrayView[Int],
) -> (Array[Int], Array[Int]) {
  let key = Array::make(65536, 0x7fc1)
  let value = Array::make(65536, 0x7fc1)
  for token in 0..<case_.positions.length() {
    let row = kv_case_row(case_, token)
    let position = case_.positions[token]
    let cell = case_.page_offsets[row] + position / 16
    let page = case_.pages[cell]
    let cache_base = page * 1024 + position % 16 * 64
    let qkv_base = token * 256
    for component in 0..<64 {
      key[cache_base + component] = qkv[qkv_base + 128 + component]
      value[cache_base + component] = qkv[qkv_base + 192 + component]
    }
  }
  (key, value)
}

///|
fn kv_compare_case(
  case_ : KvWriteCase,
  qkv : ArrayView[Int],
  key_bytes : Bytes,
  value_bytes : Bytes,
) -> (Int64, Int64) raise {
  let key = kv_bf16_bits(key_bytes)
  let value = kv_bf16_bits(value_bytes)
  let expected = kv_scalar_expected(case_, qkv)
  require_kv_probe(
    key.length() == 65536 && value.length() == 65536,
    "cache readback size drifted",
  )
  let mut key_mutations = 0L
  let mut value_mutations = 0L
  for index in 0..<65536 {
    require_kv_probe(
      key[index] == expected.0[index],
      "key cache scalar oracle mismatch",
    )
    require_kv_probe(
      value[index] == expected.1[index],
      "value cache scalar oracle mismatch",
    )
    if key[index] != 0x7fc1 {
      key_mutations += 1L
    }
    if value[index] != 0x7fc1 {
      value_mutations += 1L
    }
  }
  let expected_mutations = case_.positions.length().to_int64() * 64L
  require_kv_probe(
    key_mutations == expected_mutations && value_mutations == expected_mutations,
    "exact cache mutation count mismatch",
  )
  (key_mutations, value_mutations)
}

///|
fn kv_candidate_dimensions(
  candidate : @luna_cuda_paged_kv_write_aot.PagedKvWriteCudaCandidate,
) -> @device.KernelLaunchDimensions raise {
  let value = candidate.launch_dimensions()
  @device.KernelLaunchDimensions::new(
    grid_x=value.grid_x(),
    grid_y=value.grid_y(),
    grid_z=value.grid_z(),
    block_x=value.block_x(),
    block_y=value.block_y(),
    block_z=value.block_z(),
    shared_memory_bytes=value.shared_memory_bytes(),
  )
}

///|
fn kv_oracle_dimensions() -> @device.KernelLaunchDimensions raise {
  @device.KernelLaunchDimensions::new(
    grid_x=1,
    grid_y=1,
    grid_z=1,
    block_x=1,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
}

///|
fn run_kv_write_device_probe(
  candidate : @luna_cuda_paged_kv_write_aot.PagedKvWriteCudaCandidate,
  compiled : @luna_cuda_paged_kv_write_aot.PagedKvWriteCompiledArtifact,
  oracle_cubin : Bytes,
  cycles : Int,
  balance : KvProbeBalance,
) -> KvProbeObservation raise {
  let (cuda_driver_version, devices) = match @device.probe() {
    Available(driver_version~, devices~) => (driver_version, devices)
    Unavailable(_) => raise KvWriteProbeRejected("CUDA driver unavailable")
  }
  require_kv_probe(devices.length() > 0, "no CUDA device")
  let capability = devices[0].capability()
  require_kv_probe(
    capability.compute_major() == 12 &&
    capability.compute_minor() == 0 &&
    capability.supports_bf16(),
    "device 0 is not exact BF16 sm120",
  )
  let context = @device.open_context(0)
  balance.contexts += 1
  defer (try context.close() catch {
    _ => balance.cleanup += 1
  } noraise {
    _ => balance.contexts -= 1
  })
  let stream = context.create_stream()
  balance.streams += 1
  defer (try stream.close() catch {
    _ => balance.cleanup += 1
  } noraise {
    _ => balance.streams -= 1
  })
  let arena = context.allocate(KV_ARENA_BYTES)
  balance.allocations += 1
  balance.device_bytes += KV_ARENA_BYTES
  defer (try arena.close() catch {
    _ => balance.cleanup += 1
  } noraise {
    _ => {
      balance.allocations -= 1
      balance.device_bytes -= KV_ARENA_BYTES
    }
  })
  let candidate_module = context.load_module(compiled.module_bytes())
  balance.modules += 1
  defer (try candidate_module.close() catch {
    _ => balance.cleanup += 1
  } noraise {
    _ => balance.modules -= 1
  })
  let candidate_function = candidate_module.load_function(
    @utf8.encode(candidate.function_symbol()),
  )
  balance.functions += 1
  defer (try candidate_function.close() catch {
    _ => balance.cleanup += 1
  } noraise {
    _ => balance.functions -= 1
  })
  let oracle_module = context.load_module(oracle_cubin)
  balance.modules += 1
  defer (try oracle_module.close() catch {
    _ => balance.cleanup += 1
  } noraise {
    _ => balance.modules -= 1
  })
  let oracle_function = oracle_module.load_function(
    @utf8.encode(KV_SERIAL_ORACLE_SYMBOL),
  )
  balance.functions += 1
  defer (try oracle_function.close() catch {
    _ => balance.cleanup += 1
  } noraise {
    _ => balance.functions -= 1
  })
  let cases = kv_probe_cases()
  let mut token_count = 0
  let mut key_mutations = 0L
  let mut value_mutations = 0L
  let mut canary = 0U
  let mut launches = 0
  for _cycle in 0..<cycles {
    for case_ in cases {
      let qkv = kv_reset_case(arena, case_)
      kv_launch(
        oracle_function,
        kv_oracle_dimensions(),
        kv_oracle_arguments(arena),
        stream,
        balance,
      )
      require_kv_probe(
        kv_guards_unchanged(arena),
        "serial oracle changed a guard",
      )
      let oracle_key = arena.copy_to_host(KV_KEY, KV_CACHE_BYTES)
      let oracle_value = arena.copy_to_host(KV_VALUE, KV_CACHE_BYTES)
      ignore(kv_compare_case(case_, qkv[:], oracle_key, oracle_value))
      kv_reset_outputs(arena)
      kv_launch(
        candidate_function,
        kv_candidate_dimensions(candidate),
        kv_candidate_arguments(arena),
        stream,
        balance,
      )
      require_kv_probe(kv_guards_unchanged(arena), "candidate changed a guard")
      let candidate_key = arena.copy_to_host(KV_KEY, KV_CACHE_BYTES)
      let candidate_value = arena.copy_to_host(KV_VALUE, KV_CACHE_BYTES)
      require_kv_probe(
        candidate_key == oracle_key && candidate_value == oracle_value,
        "candidate and serial CUDA oracle differ",
      )
      let mutations = kv_compare_case(
        case_,
        qkv[:],
        candidate_key,
        candidate_value,
      )
      let observed_canary = kv_u32(arena.copy_to_host(KV_CANARY, 4L))
      require_kv_probe(
        observed_canary ==
        candidate.dispatch_canary_per_token() *
        case_.positions.length().reinterpret_as_uint(),
        "candidate dispatch canary mismatch",
      )
      if _cycle == 0 {
        token_count += case_.positions.length()
        key_mutations += mutations.0
        value_mutations += mutations.1
        canary += observed_canary
      }
      launches += 2
    }
  }
  {
    token_count,
    key_mutations,
    value_mutations,
    observed_canary: canary,
    launches,
    device_uuid: devices[0].uuid(),
    device_pci: devices[0].pci(),
    device_name_sha256: @crypto.bytes_to_hex_string(
      @crypto.sha256(@utf8.encode(devices[0].name())),
    ),
    device_total_memory_bytes: capability.total_memory_bytes(),
    cuda_driver_version,
  }
}
