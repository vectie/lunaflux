///|
const REUSABLE_FUSED_RUNTIME_MAXIMUM_MODULES : Int32 = 9

///|
const REUSABLE_FUSED_RUNTIME_MAXIMUM_MODULE_BYTES : Int32 = 8388608

///|
const REUSABLE_FUSED_RUNTIME_MAXIMUM_TOTAL_MODULE_BYTES : Int32 = 40000000

///|
fn reusable_runtime_abi_name(abi : ReusableFusedRuntimeKernelAbi) -> String {
  match abi {
    ResidualRmsNormProductionV2 => "residual-rmsnorm-production-v2"
    QwenQkvQkNormPositionedRopePagedKvWriteProductionV4 =>
      "qwen-qkv-qknorm-positioned-rope-paged-kvwrite-production-v4"
    AttentionIngressPreparedRotaryProductionV11 =>
      "attention-ingress-prepared-rotary-production-v11"
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
    PagedAttentionReadOnlyCompilerDecodeProductionV7 =>
      "paged-attention-readonly-compiler-decode-production-v7"
    PagedAttentionReadOnlyBlockwiseF32DecodeProductionV8 =>
      "paged-attention-readonly-blockwise-f32-decode-production-v8"
    PagedAttentionReadOnlyDualScoreBlockwiseF32DecodeProductionV10 =>
      "paged-attention-readonly-dual-score-blockwise-f32-decode-production-v10"
    PagedAttentionReadOnlyBlockwiseFmaF32DecodeProductionV9 =>
      "paged-attention-readonly-blockwise-fma-f32-decode-production-v9"
    PagedAttentionReadOnlyExtendedDecodeProductionV11 =>
      "paged-attention-readonly-extended-decode-production-v11"
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
    "attention-ingress-prepared-rotary-production-v11" =>
      AttentionIngressPreparedRotaryProductionV11
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
    "paged-attention-readonly-compiler-decode-production-v7" =>
      PagedAttentionReadOnlyCompilerDecodeProductionV7
    "paged-attention-readonly-blockwise-f32-decode-production-v8" =>
      PagedAttentionReadOnlyBlockwiseF32DecodeProductionV8
    "paged-attention-readonly-dual-score-blockwise-f32-decode-production-v10" =>
      PagedAttentionReadOnlyDualScoreBlockwiseF32DecodeProductionV10
    "paged-attention-readonly-blockwise-fma-f32-decode-production-v9" =>
      PagedAttentionReadOnlyBlockwiseFmaF32DecodeProductionV9
    "paged-attention-readonly-extended-decode-production-v11" =>
      PagedAttentionReadOnlyExtendedDecodeProductionV11
    _ => raise InvalidFusedArtifact(Family)
  }
}

///|
fn reusable_runtime_required_operation_count(
  abi : ReusableFusedRuntimeKernelAbi,
) -> Int32 {
  match abi {
    ResidualRmsNormProductionV2 => 2
    QwenQkvQkNormPositionedRopePagedKvWriteProductionV4 => 4
    AttentionIngressPreparedRotaryProductionV11 => 4
    QwenQkNormPositionedRopePagedKvWriteProductionV2 => 3
    QwenPagedAttentionPrefillPartitionedPartialProductionV4 => 1
    QwenPagedAttentionPrefillPartitionedMergeProductionV4 => 1
    QwenPagedAttentionPrefillPartitioned8PartialProductionV6 => 1
    QwenPagedAttentionPrefillPartitioned8MergeProductionV6 => 1
    QwenPagedAttentionPrefillWideQueryProductionV5 => 1
    QwenPagedAttentionReadOnlyProductionV2 => 1
    QwenPagedAttentionReadOnlyDecodeSplitProductionV3 => 1
    PagedAttentionReadOnlyCompilerDecodeProductionV7 => 1
    PagedAttentionReadOnlyBlockwiseF32DecodeProductionV8 => 1
    PagedAttentionReadOnlyBlockwiseFmaF32DecodeProductionV9 => 1
    PagedAttentionReadOnlyDualScoreBlockwiseF32DecodeProductionV10 => 1
    PagedAttentionReadOnlyExtendedDecodeProductionV11 => 1
  }
}

///|
fn reusable_runtime_abi_rank(abi : ReusableFusedRuntimeKernelAbi) -> Int32 {
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
    PagedAttentionReadOnlyCompilerDecodeProductionV7 => 10
    PagedAttentionReadOnlyBlockwiseF32DecodeProductionV8 => 11
    PagedAttentionReadOnlyBlockwiseFmaF32DecodeProductionV9 => 12
    PagedAttentionReadOnlyDualScoreBlockwiseF32DecodeProductionV10 => 13
    AttentionIngressPreparedRotaryProductionV11 => 14
    PagedAttentionReadOnlyExtendedDecodeProductionV11 => 15
  }
}

///|
fn reusable_runtime_spans_overlap(
  first_a : Int32,
  count_a : Int32,
  first_b : Int32,
  count_b : Int32,
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
  abi == QwenPagedAttentionReadOnlyDecodeSplitProductionV3 ||
  abi == PagedAttentionReadOnlyCompilerDecodeProductionV7 ||
  abi == PagedAttentionReadOnlyBlockwiseF32DecodeProductionV8 ||
  abi == PagedAttentionReadOnlyBlockwiseFmaF32DecodeProductionV9 ||
  abi == PagedAttentionReadOnlyDualScoreBlockwiseF32DecodeProductionV10 ||
  abi == PagedAttentionReadOnlyExtendedDecodeProductionV11
}

///|
fn reusable_runtime_is_decode(abi : ReusableFusedRuntimeKernelAbi) -> Bool {
  abi
  is (QwenPagedAttentionReadOnlyDecodeSplitProductionV3
  | PagedAttentionReadOnlyCompilerDecodeProductionV7
  | PagedAttentionReadOnlyBlockwiseF32DecodeProductionV8
  | PagedAttentionReadOnlyBlockwiseFmaF32DecodeProductionV9
  | PagedAttentionReadOnlyDualScoreBlockwiseF32DecodeProductionV10
  | PagedAttentionReadOnlyExtendedDecodeProductionV11)
}

///|
fn reusable_runtime_qwen_ingress_terminal_offset(
  abi : ReusableFusedRuntimeKernelAbi,
) -> Int32? {
  match abi {
    QwenQkvQkNormPositionedRopePagedKvWriteProductionV4 => Some(3)
    AttentionIngressPreparedRotaryProductionV11 => Some(3)
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
  first_a : Int32,
  abi_b : ReusableFusedRuntimeKernelAbi,
  first_b : Int32,
) -> Bool {
  let reader_a = reusable_runtime_is_qwen_attention_reader(abi_a)
  let reader_b = reusable_runtime_is_qwen_attention_reader(abi_b)
  if reader_a && reader_b {
    if reusable_runtime_is_decode(abi_a) && reusable_runtime_is_decode(abi_b) {
      return false
    }
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
  first_operation : Int32,
  operation_count : Int32,
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
  first_operation~ : Int32,
  operation_count~ : Int32,
  function_symbol~ : String,
  launch~ : @launch_contract.AotLaunchDimensions,
  module_bytes~ : Bytes,
  query_tile_rows? : Int32,
  query_metadata_version~ : Int32 = 0,
  heads_per_cta~ : Int32 = 1,
  decode_execution? : DecodeExecutionDescriptor,
  prefill_exponential_law~ : PrefillExponentialLaw = StrictNaturalExponential,
  allow_approximate_prefill_exponential~ : Bool = false,
  row_variant? : ReusableFusedRuntimeRowVariant,
  module_label_sha256? : String,
) -> ReusableFusedRuntimeModuleInput raise FusedArtifactAdmissionError {
  reusable_runtime_validate_prefill_exponential(
    abi, prefill_exponential_law, function_symbol, allow_approximate_prefill_exponential,
  )
  let decode_execution = reusable_runtime_decode_execution(
    abi, decode_execution,
  )
  if abi == PagedAttentionReadOnlyExtendedDecodeProductionV11 &&
    decode_execution is Some(descriptor) &&
    !descriptor.matches_function_symbol(function_symbol) {
    raise InvalidFusedArtifact(KernelAbi)
  }
  reusable_runtime_validate_head_ownership(abi, heads_per_cta, query_tile_rows)
  if query_metadata_version != 0 &&
    (
      query_metadata_version != 1 ||
      !(query_tile_rows is Some(16 | 32 | 64 | 128)) ||
      !reusable_runtime_is_qwen_attention_reader(abi)
    ) {
    raise InvalidFusedArtifact(KernelAbi)
  }
  if query_tile_rows is Some(rows) && (rows <= 0 || rows > 1048576) {
    raise InvalidFusedArtifact(Launch)
  }
  reusable_runtime_validate_module(
    abi, first_operation, operation_count, function_symbol, launch, module_bytes,
  )
  reusable_runtime_validate_row_variant(
    abi, function_symbol, launch, heads_per_cta, row_variant,
  )
  // New AOT construction may name code once. Already-labelled producer inputs
  // skip that work; packaging and route planning never rehash the code.
  let module_label = match module_label_sha256 {
    Some(label) => fused_deployment_digest(label)
    None => fused_artifact_digest(module_bytes)
  }
  {
    abi,
    first_operation,
    operation_count,
    symbol: function_symbol,
    launch,
    query_tile_rows,
    query_metadata_version,
    heads_per_cta,
    decode_execution,
    prefill_exponential_law,
    module_bytes,
    row_variant,
    module_label,
  }
}
