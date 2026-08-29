///|
struct FusedProductionProbeObservation {
  maximum_absolute_error_ppb : Int64
  maximum_relative_error_ppb : Int64
  launches : Int
  device_uuid : String
  device_pci : String
  device_name_sha256 : String
  device_total_memory_bytes : Int64
  cuda_driver_version : Int
}

///|
fn run_fused_production_device_probe(
  bindings : FusedProductionProbeBindings,
  oracle_qkv_cubin : Bytes,
  oracle_rope_cubin : Bytes,
  oracle_attention_cubin : Bytes,
  candidate_context : FusedProbeCandidateContext,
  candidates : FusedProductionProbeCandidates,
  cycles : Int,
  balance : FusedProbeResourceBalance,
) -> FusedProductionProbeObservation raise {
  let (cuda_driver_version, devices) = match @device.probe() {
    Available(driver_version~, devices~) => (driver_version, devices)
    Unavailable(_) => raise FusedCudaProbeRejected("CUDA driver is unavailable")
  }
  require_fused_probe(devices.length() > 0, "no CUDA device is available")
  let capability = devices[0].capability()
  require_fused_probe(
    capability.compute_major() == 12 && capability.compute_minor() == 0,
    "fused production V2 qualification requires exact sm120 device 0",
  )
  require_fused_probe(capability.supports_bf16(), "device 0 lacks BF16 support")
  let shapes = @fused_parallel_qualification.declared_qualification_shapes()
  require_exact_fused_qualification_shapes(shapes)
  let context = @device.open_context(0)
  balance.contexts += 1
  defer {
    context.close() catch {
      _ => balance.cleanup_failures += 1
    }
    balance.contexts -= 1
  }
  let stream = context.create_stream()
  balance.streams += 1
  defer {
    stream.close() catch {
      _ => balance.cleanup_failures += 1
    }
    balance.streams -= 1
  }
  let arena = context.allocate(PROBE_ARENA_BYTES)
  balance.allocations += 1
  balance.device_bytes += PROBE_ARENA_BYTES
  defer {
    arena.close() catch {
      _ => balance.cleanup_failures += 1
    }
    balance.allocations -= 1
    balance.device_bytes -= PROBE_ARENA_BYTES
  }
  let production_qkv = load_fused_production_kernel(
    context,
    bindings.qkv.module_bytes(),
    candidates.qkv.function_symbol(),
    balance,
  )
  defer close_fused_production_kernel(production_qkv, balance)
  let production_readonly = load_fused_production_kernel(
    context,
    bindings.readonly_module,
    candidates.readonly_candidate.function_symbol(),
    balance,
  )
  defer close_fused_production_kernel(production_readonly, balance)
  let production_residual = load_fused_production_kernel(
    context,
    bindings.residual.module_bytes(),
    candidates.residual.function_symbol(),
    balance,
  )
  defer close_fused_production_kernel(production_residual, balance)
  let oracle_qkv = load_fused_production_kernel(
    context,
    oracle_qkv_cubin,
    candidate_context.qualification.qkv_fallback().function_symbol(),
    balance,
  )
  defer close_fused_production_kernel(oracle_qkv, balance)
  let oracle_rope = load_fused_production_kernel(
    context,
    oracle_rope_cubin,
    candidate_context.qualification.rope_fallback().function_symbol(),
    balance,
  )
  defer close_fused_production_kernel(oracle_rope, balance)
  let oracle_attention = load_fused_production_kernel(
    context,
    oracle_attention_cubin,
    candidates.oracle_attention.function_symbol(),
    balance,
  )
  defer close_fused_production_kernel(oracle_attention, balance)
  let mut maximum_absolute_error_ppb = 0L
  let mut maximum_relative_error_ppb = 0L
  let mut launches = 0
  for _ in 0..<cycles {
    for shape in shapes {
      let attention = run_production_qkv_readonly_shape(
        context, stream, arena, production_qkv, production_readonly, oracle_qkv,
        oracle_rope, oracle_attention, candidate_context, candidates, shape, balance,
      )
      let residual = run_production_residual_shape(
        context,
        stream,
        arena,
        production_residual,
        candidates.residual,
        shape,
        balance,
      )
      maximum_absolute_error_ppb = maximum_absolute_error_ppb
        .max(attention.0)
        .max(residual.0)
      maximum_relative_error_ppb = maximum_relative_error_ppb
        .max(attention.1)
        .max(residual.1)
      launches += 6
    }
  }
  {
    maximum_absolute_error_ppb,
    maximum_relative_error_ppb,
    launches,
    device_uuid: devices[0].uuid(),
    device_pci: devices[0].pci(),
    device_name_sha256: campaign_sha256(@utf8.encode(devices[0].name())),
    device_total_memory_bytes: capability.total_memory_bytes(),
    cuda_driver_version,
  }
}
