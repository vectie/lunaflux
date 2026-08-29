///|
struct FusedProbeObservation {
  maximum_qkv_absolute_error_ppb : Int64
  maximum_qkv_relative_error_ppb : Int64
  qkv_dispatch_canary : UInt
  maximum_residual_absolute_error_ppb : Int64
  maximum_residual_relative_error_ppb : Int64
  residual_dispatch_canary : UInt
  launches : Int
  device_uuid : String
  device_pci : String
  device_name_sha256 : String
  device_total_memory_bytes : Int64
  cuda_driver_version : Int
}

///|
fn probe_qkv_comparison(
  actual : ArrayView[Int],
  expected : ArrayView[Int],
) -> (Int64, Int64) raise {
  let summary = @fused_parallel_qualification.compare_qkv_bf16_output(
    actual, expected,
  )
  require_fused_probe(
    @fused_parallel_qualification.qualify_qkv_bf16_output(actual, expected) ==
    summary.maximum_absolute_error(),
    "fused QKV legacy and typed referee summaries drifted",
  )
  (summary.maximum_absolute_error_ppb(), summary.maximum_relative_error_ppb())
}

///|
fn probe_residual_comparison(
  actual : ArrayView[Int],
  expected : ArrayView[Int],
) -> (Int64, Int64) raise {
  let summary = @fused_parallel_qualification.compare_residual_norm_bf16_output(
    actual, expected,
  )
  require_fused_probe(
    @fused_parallel_qualification.qualify_residual_norm_bf16_output(
      actual, expected,
    ) ==
    summary.maximum_absolute_error(),
    "fused residual legacy and typed referee summaries drifted",
  )
  (summary.maximum_absolute_error_ppb(), summary.maximum_relative_error_ppb())
}

///|
fn probe_dimensions(
  dimensions : @launch_contract.AotLaunchDimensions,
) -> @device.KernelLaunchDimensions raise {
  @device.KernelLaunchDimensions::new(
    grid_x=dimensions.grid_x(),
    grid_y=dimensions.grid_y(),
    grid_z=dimensions.grid_z(),
    block_x=dimensions.block_x(),
    block_y=dimensions.block_y(),
    block_z=dimensions.block_z(),
    shared_memory_bytes=dimensions.shared_memory_bytes(),
  )
}

///|
fn upload_qkv_fixture(
  arena : @device.Allocation,
  fixture : @fused_parallel_qualification.FusedQkvQualificationFixture,
) -> Unit raise {
  let shape = fixture.shape()
  let poison = @fused_parallel_qualification.qualification_bf16_bytes(
    Array::make(
      @fused_parallel_qualification.qualification_cache_elements(),
      @fused_parallel_qualification.qualification_poison_bf16(),
    ),
  )
  probe_upload(arena, PROBE_STEP_COUNTS, probe_i32_bytes(shape.step_counts()))
  probe_upload(arena, PROBE_QUERY_POSITIONS, probe_i32_bytes(shape.positions()))
  probe_upload(
    arena,
    PROBE_QUERY_ROW_OFFSETS,
    probe_i32_bytes(shape.query_row_offsets()),
  )
  probe_upload(
    arena,
    PROBE_SEQUENCE_LENGTHS,
    probe_i32_bytes(shape.sequence_lengths()),
  )
  probe_upload(
    arena,
    PROBE_PAGE_TABLE_OFFSETS,
    probe_i32_bytes(shape.page_table_offsets()),
  )
  probe_upload(
    arena,
    PROBE_PAGE_TABLE_INDICES,
    probe_i32_bytes(shape.page_table()),
  )
  probe_upload(
    arena,
    PROBE_INPUT,
    @fused_parallel_qualification.qualification_bf16_bytes(fixture.input()),
  )
  probe_upload(
    arena,
    PROBE_Q_WEIGHT,
    @fused_parallel_qualification.qualification_bf16_bytes(fixture.q_weight()),
  )
  probe_upload(
    arena,
    PROBE_K_WEIGHT,
    @fused_parallel_qualification.qualification_bf16_bytes(fixture.k_weight()),
  )
  probe_upload(
    arena,
    PROBE_V_WEIGHT,
    @fused_parallel_qualification.qualification_bf16_bytes(fixture.v_weight()),
  )
  probe_upload(arena, PROBE_KEY_CACHE, poison)
  probe_upload(arena, PROBE_VALUE_CACHE, poison)
  probe_upload(arena, PROBE_QKV_CANARY, probe_zero_u32())
}

///|
fn upload_residual_fixture(
  arena : @device.Allocation,
  fixture : @fused_parallel_qualification.FusedResidualNormQualificationFixture,
) -> Unit raise {
  let shape = fixture.shape()
  probe_upload(arena, PROBE_STEP_COUNTS, probe_i32_bytes(shape.step_counts()))
  probe_upload(
    arena,
    PROBE_RESIDUAL,
    @fused_parallel_qualification.qualification_bf16_bytes(fixture.residual()),
  )
  probe_upload(
    arena,
    PROBE_BRANCH,
    @fused_parallel_qualification.qualification_bf16_bytes(fixture.branch()),
  )
  probe_upload(
    arena,
    PROBE_NORM_WEIGHT,
    @fused_parallel_qualification.qualification_bf16_bytes(
      fixture.norm_weight(),
    ),
  )
  probe_upload(arena, PROBE_RESIDUAL_CANARY, probe_zero_u32())
}

///|
fn run_qkv_shape(
  function_ : @device.Function,
  stream : @device.Stream,
  arena : @device.Allocation,
  candidate : @luna_cuda_fused_parallel_aot.FusedQkvRopeKvWriteCudaCandidate,
  shape : @fused_parallel_qualification.FusedQualificationShape,
  balance : FusedProbeResourceBalance,
) -> (Int64, Int64, UInt) raise {
  let fixture = @fused_parallel_qualification.qkv_qualification_fixture(shape)
  upload_qkv_fixture(arena, fixture)
  probe_launch_synchronous(
    function_,
    probe_dimensions(candidate.launch_dimensions()),
    probe_qkv_arguments(arena, shape),
    stream,
    balance,
  )
  let observed_canary = probe_u32(arena.copy_to_host(PROBE_QKV_CANARY, 4L))
  require_fused_probe(
    observed_canary ==
    shape.expected_dispatch_canary(candidate.dispatch_canary()),
    "fused QKV dispatch canary mismatch",
  )
  let rotated = probe_qkv_comparison(
    probe_read_bf16(
      arena,
      PROBE_ROTATED_QKV,
      shape.token_count() *
      @fused_parallel_qualification.qualification_output_width(),
    ),
    fixture.expected_rotated_qkv(),
  )
  let key = probe_qkv_comparison(
    probe_read_bf16(
      arena,
      PROBE_KEY_CACHE,
      @fused_parallel_qualification.qualification_cache_elements(),
    ),
    fixture.expected_key_cache(),
  )
  let value = probe_qkv_comparison(
    probe_read_bf16(
      arena,
      PROBE_VALUE_CACHE,
      @fused_parallel_qualification.qualification_cache_elements(),
    ),
    fixture.expected_value_cache(),
  )
  (
    rotated.0.max(key.0).max(value.0),
    rotated.1.max(key.1).max(value.1),
    observed_canary,
  )
}

///|
fn run_residual_shape(
  function_ : @device.Function,
  stream : @device.Stream,
  arena : @device.Allocation,
  candidate : @luna_cuda_fused_parallel_aot.FusedResidualRmsNormCudaCandidate,
  shape : @fused_parallel_qualification.FusedQualificationShape,
  balance : FusedProbeResourceBalance,
) -> (Int64, Int64, UInt) raise {
  let fixture = @fused_parallel_qualification.residual_norm_qualification_fixture(
    shape,
  )
  upload_residual_fixture(arena, fixture)
  probe_launch_synchronous(
    function_,
    probe_dimensions(candidate.launch_dimensions()),
    probe_residual_arguments(arena, shape),
    stream,
    balance,
  )
  let observed_canary = probe_u32(arena.copy_to_host(PROBE_RESIDUAL_CANARY, 4L))
  require_fused_probe(
    observed_canary ==
    shape.expected_dispatch_canary(candidate.dispatch_canary()),
    "fused residual/RMSNorm dispatch canary mismatch",
  )
  let elements = shape.token_count() *
    @fused_parallel_qualification.qualification_input_width()
  let residual = probe_residual_comparison(
    probe_read_bf16(arena, PROBE_RESIDUAL_OUTPUT, elements),
    fixture.expected_residual_output(),
  )
  let norm = probe_residual_comparison(
    probe_read_bf16(arena, PROBE_NORM_OUTPUT, elements),
    fixture.expected_norm_output(),
  )
  (residual.0.max(norm.0), residual.1.max(norm.1), observed_canary)
}

///|
fn run_standalone_qkv_shape(
  qkv_function : @device.Function,
  rope_function : @device.Function,
  kv_write_function : @device.Function,
  stream : @device.Stream,
  arena : @device.Allocation,
  set : @luna_fused_candidate_export.LunaFusedCandidateQualificationSet,
  shape : @fused_parallel_qualification.FusedQualificationShape,
  balance : FusedProbeResourceBalance,
) -> (Int64, Int64) raise {
  let fixture = @fused_parallel_qualification.qkv_qualification_fixture(shape)
  upload_qkv_fixture(arena, fixture)
  probe_launch_synchronous(
    qkv_function,
    probe_dimensions(set.qkv_fallback().launch_dimensions()),
    probe_standalone_projection_arguments(arena, shape),
    stream,
    balance,
  )
  probe_launch_synchronous(
    rope_function,
    probe_dimensions(set.rope_fallback().launch_dimensions()),
    probe_standalone_rope_arguments(arena, shape),
    stream,
    balance,
  )
  probe_launch_synchronous(
    kv_write_function,
    probe_dimensions(set.kv_write_fallback().launch_dimensions()),
    probe_standalone_kv_write_arguments(arena, shape),
    stream,
    balance,
  )
  let rotated = probe_qkv_comparison(
    probe_read_bf16(
      arena,
      PROBE_ROTATED_QKV,
      shape.token_count() *
      @fused_parallel_qualification.qualification_output_width(),
    ),
    fixture.expected_rotated_qkv(),
  )
  let key = probe_qkv_comparison(
    probe_read_bf16(
      arena,
      PROBE_KEY_CACHE,
      @fused_parallel_qualification.qualification_cache_elements(),
    ),
    fixture.expected_key_cache(),
  )
  let value = probe_qkv_comparison(
    probe_read_bf16(
      arena,
      PROBE_VALUE_CACHE,
      @fused_parallel_qualification.qualification_cache_elements(),
    ),
    fixture.expected_value_cache(),
  )
  (rotated.0.max(key.0).max(value.0), rotated.1.max(key.1).max(value.1))
}

///|
fn run_fused_device_probe(
  bindings : FusedProbeBindings,
  oracle_modules : FusedProbeOracleModules,
  set : @luna_fused_candidate_export.LunaFusedCandidateQualificationSet,
  cycles : Int,
  balance : FusedProbeResourceBalance,
) -> FusedProbeObservation raise {
  let (cuda_driver_version, devices) = match @device.probe() {
    Available(driver_version~, devices~) => (driver_version, devices)
    Unavailable(_) => raise FusedCudaProbeRejected("CUDA driver is unavailable")
  }
  require_fused_probe(devices.length() > 0, "no CUDA device is available")
  let capability = devices[0].capability()
  require_fused_probe(
    capability.compute_major() == 12 && capability.compute_minor() == 0,
    "fused qualification requires exact sm120 device 0",
  )
  require_fused_probe(capability.supports_bf16(), "device 0 lacks BF16 support")
  let shapes = @fused_parallel_qualification.declared_qualification_shapes()
  require_exact_fused_qualification_shapes(shapes)
  let context = @device.open_context(0)
  balance.contexts += 1
  defer (try context.close() catch {
    _ => balance.cleanup_failures += 1
  } noraise {
    _ => balance.contexts -= 1
  })
  let stream = context.create_stream()
  balance.streams += 1
  defer (try stream.close() catch {
    _ => balance.cleanup_failures += 1
  } noraise {
    _ => balance.streams -= 1
  })
  let arena = context.allocate(PROBE_ARENA_BYTES)
  balance.allocations += 1
  balance.device_bytes += PROBE_ARENA_BYTES
  defer (try arena.close() catch {
    _ => balance.cleanup_failures += 1
  } noraise {
    _ => {
      balance.allocations -= 1
      balance.device_bytes -= PROBE_ARENA_BYTES
    }
  })
  let qkv_module = context.load_module(bindings.qkv.module_bytes())
  balance.modules += 1
  defer (try qkv_module.close() catch {
    _ => balance.cleanup_failures += 1
  } noraise {
    _ => balance.modules -= 1
  })
  let qkv_function = qkv_module.load_function(
    @utf8.encode(bindings.qkv.function_symbol()),
  )
  balance.functions += 1
  defer (try qkv_function.close() catch {
    _ => balance.cleanup_failures += 1
  } noraise {
    _ => balance.functions -= 1
  })
  let residual_module = context.load_module(bindings.residual.module_bytes())
  balance.modules += 1
  defer (try residual_module.close() catch {
    _ => balance.cleanup_failures += 1
  } noraise {
    _ => balance.modules -= 1
  })
  let residual_function = residual_module.load_function(
    @utf8.encode(bindings.residual.function_symbol()),
  )
  balance.functions += 1
  defer (try residual_function.close() catch {
    _ => balance.cleanup_failures += 1
  } noraise {
    _ => balance.functions -= 1
  })
  let oracle_qkv_module = context.load_module(oracle_modules.qkv)
  balance.modules += 1
  defer (try oracle_qkv_module.close() catch {
    _ => balance.cleanup_failures += 1
  } noraise {
    _ => balance.modules -= 1
  })
  let oracle_qkv_function = oracle_qkv_module.load_function(
    @utf8.encode(set.qkv_fallback().function_symbol()),
  )
  balance.functions += 1
  defer (try oracle_qkv_function.close() catch {
    _ => balance.cleanup_failures += 1
  } noraise {
    _ => balance.functions -= 1
  })
  let oracle_rope_module = context.load_module(oracle_modules.rope)
  balance.modules += 1
  defer (try oracle_rope_module.close() catch {
    _ => balance.cleanup_failures += 1
  } noraise {
    _ => balance.modules -= 1
  })
  let oracle_rope_function = oracle_rope_module.load_function(
    @utf8.encode(set.rope_fallback().function_symbol()),
  )
  balance.functions += 1
  defer (try oracle_rope_function.close() catch {
    _ => balance.cleanup_failures += 1
  } noraise {
    _ => balance.functions -= 1
  })
  let oracle_kv_module = context.load_module(oracle_modules.kv_write)
  balance.modules += 1
  defer (try oracle_kv_module.close() catch {
    _ => balance.cleanup_failures += 1
  } noraise {
    _ => balance.modules -= 1
  })
  let oracle_kv_function = oracle_kv_module.load_function(
    @utf8.encode(set.kv_write_fallback().function_symbol()),
  )
  balance.functions += 1
  defer (try oracle_kv_function.close() catch {
    _ => balance.cleanup_failures += 1
  } noraise {
    _ => balance.functions -= 1
  })
  let qkv_candidate = set.qkv_candidate()
  let residual_candidate = set.residual_norm_candidate()
  let mut qkv_absolute_ppb = 0L
  let mut qkv_relative_ppb = 0L
  let mut qkv_dispatch_canary = 0U
  let mut residual_absolute_ppb = 0L
  let mut residual_relative_ppb = 0L
  let mut residual_dispatch_canary = 0U
  let mut launches = 0
  for cycle in 0..<cycles {
    for shape in shapes {
      let oracle = run_standalone_qkv_shape(
        oracle_qkv_function, oracle_rope_function, oracle_kv_function, stream, arena,
        set, shape, balance,
      )
      qkv_absolute_ppb = qkv_absolute_ppb.max(oracle.0)
      qkv_relative_ppb = qkv_relative_ppb.max(oracle.1)
      let qkv = run_qkv_shape(
        qkv_function, stream, arena, qkv_candidate, shape, balance,
      )
      qkv_absolute_ppb = qkv_absolute_ppb.max(qkv.0)
      qkv_relative_ppb = qkv_relative_ppb.max(qkv.1)
      let residual = run_residual_shape(
        residual_function, stream, arena, residual_candidate, shape, balance,
      )
      residual_absolute_ppb = residual_absolute_ppb.max(residual.0)
      residual_relative_ppb = residual_relative_ppb.max(residual.1)
      if cycle == 0 {
        qkv_dispatch_canary += qkv.2
        residual_dispatch_canary += residual.2
      }
      launches += 5
    }
  }
  {
    maximum_qkv_absolute_error_ppb: qkv_absolute_ppb,
    maximum_qkv_relative_error_ppb: qkv_relative_ppb,
    qkv_dispatch_canary,
    maximum_residual_absolute_error_ppb: residual_absolute_ppb,
    maximum_residual_relative_error_ppb: residual_relative_ppb,
    residual_dispatch_canary,
    launches,
    device_uuid: devices[0].uuid(),
    device_pci: devices[0].pci(),
    device_name_sha256: campaign_sha256(@utf8.encode(devices[0].name())),
    device_total_memory_bytes: capability.total_memory_bytes(),
    cuda_driver_version,
  }
}
