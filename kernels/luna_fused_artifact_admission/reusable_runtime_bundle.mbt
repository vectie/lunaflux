///|
const REUSABLE_FUSED_RUNTIME_MAXIMUM_MODULES : Int = 9

///|
const REUSABLE_FUSED_RUNTIME_MAXIMUM_MODULE_BYTES : Int = 8388608

///|
const REUSABLE_FUSED_RUNTIME_MAXIMUM_TOTAL_MODULE_BYTES : Int = 40000000

///|
fn reusable_runtime_abi_name(abi : ReusableFusedRuntimeKernelAbi) -> String {
  match abi {
    ResidualRmsNormProductionV2 => "residual-rmsnorm-production-v2"
    QwenQkvQkNormPositionedRopePagedKvWriteProductionV4 =>
      "qwen-qkv-qknorm-positioned-rope-paged-kvwrite-production-v4"
    QwenQkNormPositionedRopePagedKvWriteProductionV2 =>
      "qwen-qknorm-positioned-rope-paged-kvwrite-production-v2"
    QwenPagedAttentionPrefillPartitionedPartialProductionV4 =>
      "qwen-paged-attention-prefill-partitioned-partial-production-v4"
    QwenPagedAttentionPrefillPartitionedMergeProductionV4 =>
      "qwen-paged-attention-prefill-partitioned-merge-production-v4"
    QwenPagedAttentionPrefillPartitioned8PartialProductionV6 =>
      "qwen-paged-attention-prefill-partitioned8-partial-production-v6"
    QwenPagedAttentionPrefillPartitioned8MergeProductionV6 =>
      "qwen-paged-attention-prefill-partitioned8-merge-production-v6"
    QwenPagedAttentionPrefillWideQueryProductionV5 =>
      "qwen-paged-attention-prefill-wide-query-production-v5"
    QwenPagedAttentionReadOnlyProductionV2 =>
      "qwen-paged-attention-readonly-production-v2"
    QwenPagedAttentionReadOnlyDecodeSplitProductionV3 =>
      "qwen-paged-attention-readonly-decode-split-production-v3"
  }
}

///|
fn reusable_runtime_parse_abi(
  name : String,
) -> ReusableFusedRuntimeKernelAbi raise FusedArtifactAdmissionError {
  match name {
    "residual-rmsnorm-production-v2" => ResidualRmsNormProductionV2
    "qwen-qkv-qknorm-positioned-rope-paged-kvwrite-production-v4" =>
      QwenQkvQkNormPositionedRopePagedKvWriteProductionV4
    "qwen-qknorm-positioned-rope-paged-kvwrite-production-v2" =>
      QwenQkNormPositionedRopePagedKvWriteProductionV2
    "qwen-paged-attention-prefill-partitioned-partial-production-v4" =>
      QwenPagedAttentionPrefillPartitionedPartialProductionV4
    "qwen-paged-attention-prefill-partitioned-merge-production-v4" =>
      QwenPagedAttentionPrefillPartitionedMergeProductionV4
    "qwen-paged-attention-prefill-partitioned8-partial-production-v6" =>
      QwenPagedAttentionPrefillPartitioned8PartialProductionV6
    "qwen-paged-attention-prefill-partitioned8-merge-production-v6" =>
      QwenPagedAttentionPrefillPartitioned8MergeProductionV6
    "qwen-paged-attention-prefill-wide-query-production-v5" =>
      QwenPagedAttentionPrefillWideQueryProductionV5
    "qwen-paged-attention-readonly-production-v2" =>
      QwenPagedAttentionReadOnlyProductionV2
    "qwen-paged-attention-readonly-decode-split-production-v3" =>
      QwenPagedAttentionReadOnlyDecodeSplitProductionV3
    _ => raise InvalidFusedArtifact(Family)
  }
}

///|
fn reusable_runtime_required_operation_count(
  abi : ReusableFusedRuntimeKernelAbi,
) -> Int {
  match abi {
    ResidualRmsNormProductionV2 => 2
    QwenQkvQkNormPositionedRopePagedKvWriteProductionV4 => 4
    QwenQkNormPositionedRopePagedKvWriteProductionV2 => 3
    QwenPagedAttentionPrefillPartitionedPartialProductionV4 => 1
    QwenPagedAttentionPrefillPartitionedMergeProductionV4 => 1
    QwenPagedAttentionPrefillPartitioned8PartialProductionV6 => 1
    QwenPagedAttentionPrefillPartitioned8MergeProductionV6 => 1
    QwenPagedAttentionPrefillWideQueryProductionV5 => 1
    QwenPagedAttentionReadOnlyProductionV2 => 1
    QwenPagedAttentionReadOnlyDecodeSplitProductionV3 => 1
  }
}

///|
fn reusable_runtime_abi_rank(abi : ReusableFusedRuntimeKernelAbi) -> Int {
  match abi {
    ResidualRmsNormProductionV2 => 0
    QwenQkvQkNormPositionedRopePagedKvWriteProductionV4 => 1
    QwenQkNormPositionedRopePagedKvWriteProductionV2 => 2
    QwenPagedAttentionPrefillPartitionedPartialProductionV4 => 3
    QwenPagedAttentionPrefillPartitionedMergeProductionV4 => 4
    QwenPagedAttentionPrefillPartitioned8PartialProductionV6 => 5
    QwenPagedAttentionPrefillPartitioned8MergeProductionV6 => 6
    QwenPagedAttentionPrefillWideQueryProductionV5 => 7
    QwenPagedAttentionReadOnlyProductionV2 => 8
    QwenPagedAttentionReadOnlyDecodeSplitProductionV3 => 9
  }
}

///|
fn reusable_runtime_spans_overlap(
  first_a : Int,
  count_a : Int,
  first_b : Int,
  count_b : Int,
) -> Bool {
  first_a < first_b + count_b && first_b < first_a + count_a
}

///|
fn reusable_runtime_is_qwen_attention_reader(
  abi : ReusableFusedRuntimeKernelAbi,
) -> Bool {
  abi == QwenPagedAttentionPrefillPartitionedPartialProductionV4 ||
  abi == QwenPagedAttentionPrefillPartitionedMergeProductionV4 ||
  abi == QwenPagedAttentionPrefillPartitioned8PartialProductionV6 ||
  abi == QwenPagedAttentionPrefillPartitioned8MergeProductionV6 ||
  abi == QwenPagedAttentionPrefillWideQueryProductionV5 ||
  abi == QwenPagedAttentionReadOnlyProductionV2 ||
  abi == QwenPagedAttentionReadOnlyDecodeSplitProductionV3
}

///|
fn reusable_runtime_qwen_ingress_terminal_offset(
  abi : ReusableFusedRuntimeKernelAbi,
) -> Int? {
  match abi {
    QwenQkvQkNormPositionedRopePagedKvWriteProductionV4 => Some(3)
    QwenQkNormPositionedRopePagedKvWriteProductionV2 => Some(2)
    _ => None
  }
}

///|
/// The Qwen ingress fusion consumes the attention operation's KV-write role,
/// while the separate readonly kernel consumes that same operation's cache
/// read role. This terminal overlap is the only valid shared operation claim.
fn reusable_runtime_allows_terminal_qwen_attention_overlap(
  abi_a : ReusableFusedRuntimeKernelAbi,
  first_a : Int,
  abi_b : ReusableFusedRuntimeKernelAbi,
  first_b : Int,
) -> Bool {
  let reader_a = reusable_runtime_is_qwen_attention_reader(abi_a)
  let reader_b = reusable_runtime_is_qwen_attention_reader(abi_b)
  if reader_a && reader_b {
    return first_a == first_b
  }
  match
    (
      reader_a,
      reusable_runtime_qwen_ingress_terminal_offset(abi_b),
      reader_b,
      reusable_runtime_qwen_ingress_terminal_offset(abi_a),
    ) {
    (true, Some(offset), _, _) => first_a == first_b + offset
    (_, _, true, Some(offset)) => first_b == first_a + offset
    _ => false
  }
}

///|
fn reusable_runtime_validate_module(
  abi : ReusableFusedRuntimeKernelAbi,
  first_operation : Int,
  operation_count : Int,
  symbol : String,
  launch : @launch_contract.AotLaunchDimensions,
  module_bytes : Bytes,
) -> Unit raise FusedArtifactAdmissionError {
  let valid_grid_z = if abi ==
    QwenPagedAttentionPrefillPartitionedPartialProductionV4 ||
    abi == QwenPagedAttentionPrefillPartitioned8PartialProductionV6 {
    launch.grid_z() > 1
  } else {
    launch.grid_z() == 1
  }
  if first_operation < 0 ||
    operation_count != reusable_runtime_required_operation_count(abi) ||
    first_operation > 0x7fffffff - operation_count ||
    symbol.is_empty() ||
    symbol.contains("\n") ||
    symbol.contains("\r") ||
    launch.grid_x() <= 0 ||
    launch.grid_y() <= 0 ||
    !valid_grid_z ||
    launch.block_x() <= 0 ||
    launch.block_y() != 1 ||
    launch.block_z() != 1 ||
    module_bytes.is_empty() ||
    module_bytes.length() > REUSABLE_FUSED_RUNTIME_MAXIMUM_MODULE_BYTES {
    raise InvalidFusedArtifact(Module)
  }
}

///|
pub fn ReusableFusedRuntimeModuleInput::new(
  abi~ : ReusableFusedRuntimeKernelAbi,
  first_operation~ : Int,
  operation_count~ : Int,
  function_symbol~ : String,
  launch~ : @launch_contract.AotLaunchDimensions,
  module_bytes~ : Bytes,
  query_tile_rows? : Int,
) -> ReusableFusedRuntimeModuleInput raise FusedArtifactAdmissionError {
  if query_tile_rows is Some(rows) && (rows <= 0 || rows > 1048576) {
    raise InvalidFusedArtifact(Launch)
  }
  reusable_runtime_validate_module(
    abi, first_operation, operation_count, function_symbol, launch, module_bytes,
  )
  {
    abi,
    first_operation,
    operation_count,
    symbol: function_symbol,
    launch,
    query_tile_rows,
    module_bytes,
  }
}

///|
fn reusable_runtime_validate_unique_inputs(
  modules : ArrayView[ReusableFusedRuntimeModuleInput],
) -> Unit raise FusedArtifactAdmissionError {
  if modules.is_empty() ||
    modules.length() > REUSABLE_FUSED_RUNTIME_MAXIMUM_MODULES {
    raise InvalidFusedArtifact(Binding)
  }
  let mut total_bytes = 0
  for index, module_ in modules {
    reusable_runtime_validate_module(
      module_.abi,
      module_.first_operation,
      module_.operation_count,
      module_.symbol,
      module_.launch,
      module_.module_bytes,
    )
    total_bytes += module_.module_bytes.length()
    if total_bytes > REUSABLE_FUSED_RUNTIME_MAXIMUM_TOTAL_MODULE_BYTES {
      raise InvalidFusedArtifact(Module)
    }
    for prior in 0..<index {
      if modules[prior].abi == module_.abi {
        raise MismatchedFusedArtifact(Family)
      }
      if reusable_runtime_spans_overlap(
          modules[prior].first_operation,
          modules[prior].operation_count,
          module_.first_operation,
          module_.operation_count,
        ) &&
        !reusable_runtime_allows_terminal_qwen_attention_overlap(
          modules[prior].abi,
          modules[prior].first_operation,
          module_.abi,
          module_.first_operation,
        ) {
        raise MismatchedFusedArtifact(Binding)
      }
    }
  }
}

///|
fn reusable_runtime_validate_unique_approved(
  modules : ArrayView[ApprovedReusableFusedRuntimeModule],
) -> Unit raise FusedArtifactAdmissionError {
  if modules.is_empty() ||
    modules.length() > REUSABLE_FUSED_RUNTIME_MAXIMUM_MODULES {
    raise InvalidFusedArtifact(Binding)
  }
  let mut total_bytes = 0
  for index, module_ in modules {
    reusable_runtime_validate_module(
      module_.abi,
      module_.first_operation,
      module_.operation_count,
      module_.symbol,
      module_.launch,
      module_.module_bytes,
    )
    total_bytes += module_.module_bytes.length()
    if total_bytes > REUSABLE_FUSED_RUNTIME_MAXIMUM_TOTAL_MODULE_BYTES {
      raise InvalidFusedArtifact(Module)
    }
    for prior in 0..<index {
      if reusable_runtime_abi_rank(modules[prior].abi) >=
        reusable_runtime_abi_rank(module_.abi) {
        raise MismatchedFusedArtifact(Family)
      }
      if reusable_runtime_spans_overlap(
          modules[prior].first_operation,
          modules[prior].operation_count,
          module_.first_operation,
          module_.operation_count,
        ) &&
        !reusable_runtime_allows_terminal_qwen_attention_overlap(
          modules[prior].abi,
          modules[prior].first_operation,
          module_.abi,
          module_.first_operation,
        ) {
        raise MismatchedFusedArtifact(Binding)
      }
    }
  }
}

///|
fn reusable_runtime_append_target(
  out : StringBuilder,
  target : @catalog.DeviceTarget,
) -> Unit {
  out.write_string("compute_major=" + target.compute_major().to_string() + "\n")
  out.write_string("compute_minor=" + target.compute_minor().to_string() + "\n")
  out.write_string(
    "supports_bf16=" + (if target.supports_bf16() { "1" } else { "0" }) + "\n",
  )
  out.write_string(
    "supports_cublas_lt=" +
    (if target.supports_cublas_lt() { "1" } else { "0" }) +
    "\n",
  )
}

///|
fn reusable_runtime_append_module(
  out : StringBuilder,
  ordinal : Int,
  module_ : ReusableFusedRuntimeModuleInput,
) -> Unit raise FusedArtifactAdmissionError {
  let prefix = "module_" + ordinal.to_string() + "_"
  out.write_string(
    prefix + "abi=" + reusable_runtime_abi_name(module_.abi) + "\n",
  )
  out.write_string(
    prefix + "first_operation=" + module_.first_operation.to_string() + "\n",
  )
  out.write_string(
    prefix + "operation_count=" + module_.operation_count.to_string() + "\n",
  )
  out.write_string(prefix + "function_symbol=" + module_.symbol + "\n")
  let query_tile_rows = match module_.query_tile_rows {
    Some(rows) => rows
    None => 0
  }
  out.write_string(
    prefix + "query_tile_rows=" + query_tile_rows.to_string() + "\n",
  )
  out.write_string(
    prefix + "grid_x=" + module_.launch.grid_x().to_string() + "\n",
  )
  out.write_string(
    prefix + "grid_y=" + module_.launch.grid_y().to_string() + "\n",
  )
  out.write_string(
    prefix + "grid_z=" + module_.launch.grid_z().to_string() + "\n",
  )
  out.write_string(
    prefix + "block_x=" + module_.launch.block_x().to_string() + "\n",
  )
  out.write_string(
    prefix + "block_y=" + module_.launch.block_y().to_string() + "\n",
  )
  out.write_string(
    prefix + "block_z=" + module_.launch.block_z().to_string() + "\n",
  )
  out.write_string(
    prefix +
    "shared_bytes=" +
    module_.launch.shared_memory_bytes().to_string() +
    "\n",
  )
  out.write_string(
    prefix +
    "module_sha256=" +
    fused_artifact_digest(module_.module_bytes).as_hex() +
    "\n",
  )
  out.write_string(
    prefix +
    "module_hex=" +
    @crypto.bytes_to_hex_string(module_.module_bytes) +
    "\n",
  )
}

///|
/// Canonically packages a fixed-size set of heterogeneous production modules.
/// The resulting bytes remain inert until admitted against their exact SHA-256.
pub fn export_reusable_fused_runtime_bundle(
  model_content_sha256~ : String,
  model_plan_sha256~ : String,
  target~ : @catalog.DeviceTarget,
  modules~ : ArrayView[ReusableFusedRuntimeModuleInput],
) -> Bytes raise FusedArtifactAdmissionError {
  ignore(fused_deployment_digest(model_content_sha256))
  ignore(fused_deployment_digest(model_plan_sha256))
  if !target.supports_bf16() {
    raise InvalidFusedArtifact(Target)
  }
  reusable_runtime_validate_unique_inputs(modules)
  let out = StringBuilder()
  out.write_string("schema=lunaflux-reusable-fused-runtime-bundle.v4\n")
  out.write_string("model_content_sha256=" + model_content_sha256 + "\n")
  out.write_string("model_plan_sha256=" + model_plan_sha256 + "\n")
  reusable_runtime_append_target(out, target)
  out.write_string("module_count=" + modules.length().to_string() + "\n")
  let canonical_abis = [
    ResidualRmsNormProductionV2,
    QwenQkvQkNormPositionedRopePagedKvWriteProductionV4,
    QwenQkNormPositionedRopePagedKvWriteProductionV2,
    QwenPagedAttentionPrefillPartitionedPartialProductionV4,
    QwenPagedAttentionPrefillPartitionedMergeProductionV4,
    QwenPagedAttentionPrefillPartitioned8PartialProductionV6,
    QwenPagedAttentionPrefillPartitioned8MergeProductionV6,
    QwenPagedAttentionPrefillWideQueryProductionV5,
    QwenPagedAttentionReadOnlyProductionV2,
    QwenPagedAttentionReadOnlyDecodeSplitProductionV3,
  ]
  let mut ordinal = 0
  for abi in canonical_abis {
    for module_ in modules {
      if module_.abi == abi {
        reusable_runtime_append_module(out, ordinal, module_)
        ordinal += 1
      }
    }
  }
  let canonical = @utf8.encode(out.to_string())
  ignore(
    admit_digest_pinned_reusable_fused_runtime_bundle(
      canonical,
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ),
  )
  canonical
}

///|
/// Admits the bundle once at startup. Canonical parsing, SHA-256 validation,
/// duplicate rejection, and module hashing all finish before the immutable
/// typed bundle can reach a runtime owner.
pub fn admit_digest_pinned_reusable_fused_runtime_bundle(
  canonical : Bytes,
  expected_sha256 : String,
) -> ApprovedReusableFusedRuntimeBundle raise FusedArtifactAdmissionError {
  if @crypto.bytes_to_hex_string(@crypto.sha256(canonical)) != expected_sha256 {
    raise MismatchedFusedArtifact(Manifest)
  }
  let source_digest = fused_deployment_digest(expected_sha256)
  let reader = fused_deployment_reader(canonical)
  let has_schedule_metadata = match reader.value("schema") {
    "lunaflux-reusable-fused-runtime-bundle.v4" => true
    "lunaflux-reusable-fused-runtime-bundle.v3" => false
    _ => raise InvalidFusedArtifact(Manifest)
  }
  let model_content = fused_deployment_digest(
    reader.value("model_content_sha256"),
  )
  let model_plan = fused_deployment_digest(reader.value("model_plan_sha256"))
  let compute_major = reader.int("compute_major")
  let compute_minor = reader.int("compute_minor")
  let supports_bf16 = reader.int("supports_bf16")
  let supports_cublas_lt = reader.int("supports_cublas_lt")
  if (supports_bf16 != 0 && supports_bf16 != 1) ||
    (supports_cublas_lt != 0 && supports_cublas_lt != 1) {
    raise InvalidFusedArtifact(Target)
  }
  let target = @catalog.DeviceTarget::new(
    compute_major~,
    compute_minor~,
    supports_bf16=supports_bf16 == 1,
    supports_cublas_lt=supports_cublas_lt == 1,
  ) catch {
    _ => raise InvalidFusedArtifact(Target)
  }
  if !target.supports_bf16() {
    raise InvalidFusedArtifact(Target)
  }
  let module_count = reader.int("module_count")
  if module_count <= 0 || module_count > REUSABLE_FUSED_RUNTIME_MAXIMUM_MODULES {
    raise InvalidFusedArtifact(Binding)
  }
  let modules : Array[ApprovedReusableFusedRuntimeModule] = []
  for ordinal in 0..<module_count {
    let prefix = "module_" + ordinal.to_string() + "_"
    let abi = reusable_runtime_parse_abi(reader.value(prefix + "abi"))
    let first_operation = reader.int(prefix + "first_operation")
    let operation_count = reader.int(prefix + "operation_count")
    let symbol = reader.value(prefix + "function_symbol")
    let query_tile_rows = if has_schedule_metadata {
      let rows = reader.int(prefix + "query_tile_rows")
      if rows < 0 || rows > 1048576 {
        raise InvalidFusedArtifact(Launch)
      }
      if rows == 0 {
        None
      } else {
        Some(rows)
      }
    } else {
      None
    }
    let launch = @launch_contract.AotLaunchDimensions::new(
      grid_x=reader.int(prefix + "grid_x"),
      grid_y=reader.int(prefix + "grid_y"),
      grid_z=reader.int(prefix + "grid_z"),
      block_x=reader.int(prefix + "block_x"),
      block_y=reader.int(prefix + "block_y"),
      block_z=reader.int(prefix + "block_z"),
      shared_memory_bytes=reader.int(prefix + "shared_bytes"),
    ) catch {
      _ => raise InvalidFusedArtifact(Launch)
    }
    let module_digest = fused_deployment_digest(
      reader.value(prefix + "module_sha256"),
    )
    let module_bytes = fused_deployment_decode_hex(
      reader.value(prefix + "module_hex"),
    )
    reusable_runtime_validate_module(
      abi, first_operation, operation_count, symbol, launch, module_bytes,
    )
    if fused_artifact_digest(module_bytes) != module_digest {
      raise MismatchedFusedArtifact(Module)
    }
    modules.push({
      abi,
      first_operation,
      operation_count,
      symbol,
      launch,
      query_tile_rows,
      module_digest,
      module_bytes,
    })
  }
  if reader.cursor != reader.lines.length() - 1 {
    raise InvalidFusedArtifact(Manifest)
  }
  reusable_runtime_validate_unique_approved(modules)
  {
    source_digest,
    model_content,
    model_plan,
    target,
    modules: ReadOnlyArray::from_array(modules),
  }
}
