///|
fn qwen3_region(
  allocation : @device.Allocation,
  bytes : Int,
  alignment : Int64,
) -> @device.KernelArgument raise {
  @device.KernelArgument::device_region(
    allocation,
    offset=0L,
    byte_count=bytes.to_int64(),
    alignment~,
  )
}

///|
fn qwen3_dimensions(
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
fn qwen3_run_lowering_probe_inner(
  spec : @qwen3_spec.Qwen3ModelSpec,
  qkv : @luna_cuda_projection_aot.ProjectionCudaLowering,
  qk : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  qkv_cubin : Bytes,
  qk_cubin : Bytes,
  expected_uuid : String,
  expected_pci : String,
  cycles : Int,
  balance : Qwen3ProbeBalance,
) -> Float raise Qwen3PhysicalError {
  let context = @device.open_context(0) catch {
    _ => raise DeviceMismatch("CUDA context unavailable")
  }
  balance.contexts += 1
  defer (try context.close() catch {
    _ => balance.cleanup += 1
  } noraise {
    _ => balance.contexts -= 1
  })
  let identity = context.device_identity()
  qwen3_require(
    identity.uuid() == expected_uuid && identity.pci() == expected_pci,
    "probe device identity differs",
  )
  let capability = identity.capability()
  qwen3_require(
    capability.compute_major() == 12 &&
    capability.compute_minor() == 0 &&
    capability.supports_bf16(),
    "probe requires exact BF16 sm120",
  )
  let stream = context.create_stream() catch {
    _ => raise DeviceMismatch("probe stream unavailable")
  }
  balance.streams += 1
  defer (try stream.close() catch {
    _ => balance.cleanup += 1
  } noraise {
    _ => balance.streams -= 1
  })
  let hidden = spec.hidden_size()
  let query_width = spec.query_projection_width()
  let kv_width = spec.kv_projection_width()
  let output_width = query_width + 2 * kv_width
  let counts = qwen3_i32_counts()
  let input = qwen3_input_fixture(hidden)
  let query_weight = qwen3_sparse_weight(query_width, hidden, 1)
  let key_weight = qwen3_sparse_weight(kv_width, hidden, 2)
  let value_weight = qwen3_sparse_weight(kv_width, hidden, 3)
  let q_norm = qwen3_norm_weight(spec.head_dimension(), 1)
  let k_norm = qwen3_norm_weight(spec.head_dimension(), 3)
  let host = [
    counts, input, query_weight, key_weight, value_weight, q_norm, k_norm,
  ]
  let allocations : Array[(@device.Allocation, Int)] = []
  for bytes in host {
    let allocation = context.allocate(bytes.length().to_int64()) catch {
      _ => raise NumericMaterialization("probe allocation failed")
    }
    allocation.copy_from_host(bytes, 0L, 0L, bytes.length().to_int64()) catch {
      _ => raise NumericMaterialization("probe upload failed")
    }
    allocations.push((allocation, bytes.length()))
    balance.allocations += 1
  }
  for _ in 0..<2 {
    let allocation = context.allocate((output_width * 2).to_int64()) catch {
      _ => raise NumericMaterialization("probe output allocation failed")
    }
    allocations.push((allocation, output_width * 2))
    balance.allocations += 1
  }
  defer (for pair in allocations.rev() {
    try pair.0.close() catch {
      _ => balance.cleanup += 1
    } noraise {
      _ => balance.allocations -= 1
    }
  })
  let qkv_module = context.load_module(qkv_cubin) catch {
    _ => raise CandidateMismatch("QKV CUBIN load failed")
  }
  balance.modules += 1
  defer (try qkv_module.close() catch {
    _ => balance.cleanup += 1
  } noraise {
    _ => balance.modules -= 1
  })
  let qkv_function = qkv_module.load_function(
    @utf8.encode(qkv.function_symbol()),
  ) catch {
    _ => raise CandidateMismatch("QKV function load failed")
  }
  balance.functions += 1
  defer (try qkv_function.close() catch {
    _ => balance.cleanup += 1
  } noraise {
    _ => balance.functions -= 1
  })
  let qk_module = context.load_module(qk_cubin) catch {
    _ => raise CandidateMismatch("Q/K RMS CUBIN load failed")
  }
  balance.modules += 1
  defer (try qk_module.close() catch {
    _ => balance.cleanup += 1
  } noraise {
    _ => balance.modules -= 1
  })
  let qk_function = qk_module.load_function(@utf8.encode(qk.function_symbol())) catch {
    _ => raise CandidateMismatch("Q/K RMS function load failed")
  }
  balance.functions += 1
  defer (try qk_function.close() catch {
    _ => balance.cleanup += 1
  } noraise {
    _ => balance.functions -= 1
  })
  let qkv_arguments = @device.KernelArguments::new([
    qwen3_region(allocations[0].0, 20, 4L),
    qwen3_region(allocations[1].0, input.length(), 16L),
    qwen3_region(allocations[2].0, query_weight.length(), 16L),
    qwen3_region(allocations[3].0, key_weight.length(), 16L),
    qwen3_region(allocations[4].0, value_weight.length(), 16L),
    qwen3_region(allocations[7].0, output_width * 2, 16L),
  ]) catch {
    _ => raise CandidateMismatch("QKV ABI argument construction failed")
  }
  let qk_arguments = @device.KernelArguments::new([
    qwen3_region(allocations[0].0, 20, 4L),
    qwen3_region(allocations[7].0, output_width * 2, 16L),
    qwen3_region(allocations[5].0, q_norm.length(), 16L),
    qwen3_region(allocations[6].0, k_norm.length(), 16L),
    qwen3_region(allocations[8].0, output_width * 2, 16L),
  ]) catch {
    _ => raise CandidateMismatch("Q/K RMS ABI argument construction failed")
  }
  let expected_qkv = qwen3_expected_qkv(
    input, query_weight, key_weight, value_weight, hidden, query_width, kv_width,
  )
  let mut maximum_error = 0.0F
  for _ in 0..<cycles {
    qkv_function.launch_synchronous(
      qwen3_dimensions(qkv.launch_dimensions()),
      qkv_arguments,
      stream,
    ) catch {
      _ => raise CandidateMismatch("QKV launch failed")
    }
    let qkv_actual = allocations[7].0.copy_to_host(
      0L,
      (output_width * 2).to_int64(),
    ) catch {
      _ => raise CandidateMismatch("QKV readback failed")
    }
    qwen3_compare_qkv(qkv_actual, expected_qkv)
    qk_function.launch_synchronous(
      qwen3_dimensions(qk.launch_dimensions()),
      qk_arguments,
      stream,
    ) catch {
      _ => raise CandidateMismatch("Q/K RMS launch failed")
    }
    let qk_actual = allocations[8].0.copy_to_host(
      0L,
      (output_width * 2).to_int64(),
    ) catch {
      _ => raise CandidateMismatch("Q/K RMS readback failed")
    }
    maximum_error = maximum_error.max(
      qwen3_compare_qk_norm(
        qkv_actual,
        qk_actual,
        q_norm,
        k_norm,
        spec.query_head_count(),
        spec.kv_head_count(),
        spec.head_dimension(),
        Float::from_double(spec.rms_norm_epsilon()),
      ),
    )
  }
  maximum_error
}

///|
fn qwen3_run_lowering_probe(
  spec : @qwen3_spec.Qwen3ModelSpec,
  qkv : @luna_cuda_projection_aot.ProjectionCudaLowering,
  qk : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  qkv_cubin : Bytes,
  qk_cubin : Bytes,
  expected_uuid : String,
  expected_pci : String,
  cycles : Int,
) -> Float raise Qwen3PhysicalError {
  let balance = qwen3_probe_balance()
  let maximum_error = qwen3_run_lowering_probe_inner(
    spec, qkv, qk, qkv_cubin, qk_cubin, expected_uuid, expected_pci, cycles, balance,
  )
  qwen3_require(balance.is_zero(), "probe resource cleanup is incomplete")
  maximum_error
}

///|
async fn qwen3_probe(args : ArrayView[String]) -> Unit raise Qwen3PhysicalError {
  qwen3_require(args.length() == 14, "probe arguments are invalid")
  let metadata = qwen3_load_metadata(args[2], args[3], args[4])
  let inputs = qwen3_candidate_inputs(
    metadata,
    args[5],
    qwen3_parse_uint(args[6]),
    qwen3_parse_uint(args[7]),
    qwen3_parse_uint(args[8]),
  )
  guard inputs.candidates.candidates()[2] is Projection(qkv) &&
    inputs.candidates.candidates()[3] is Pointwise(qk) else {
    raise CandidateMismatch("Qwen probe candidate order drifted")
  }
  let cycles = @string.parse_int(args[13], base=10) catch {
    _ => raise InvalidInput("probe cycles are invalid")
  }
  qwen3_require(cycles > 0 && cycles <= 16, "probe cycles are outside 1..16")
  let qkv_cubin = qwen3_read_canonical(args[9], 67108864)
  let qk_cubin = qwen3_read_canonical(args[10], 67108864)
  let maximum_error = qwen3_run_lowering_probe(
    metadata.spec(),
    qkv,
    qk,
    qkv_cubin,
    qk_cubin,
    args[11],
    args[12],
    cycles,
  )
  println(
    "outcome=qwen3-bf16-lowerings-pass cycles=" +
    cycles.to_string() +
    " qkv=independent-width-ordered-f32-bf16 qk_rms_norm=capability-11" +
    " maximum_qk_abs_error=" +
    maximum_error.to_string() +
    " device_uuid=" +
    args[11] +
    " device_pci=" +
    args[12] +
    " resources=context0,stream0,allocation0,module0,function0,pending0,cleanup0" +
    " authority=qualification-only manifest_bindable=0 promotion_authority=0",
  )
}
