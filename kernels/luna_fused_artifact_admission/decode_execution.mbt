///|
/// Closed numerical permissions, independent of tuning-candidate identity.
pub(all) enum DecodeNumericLaw {
  OrderedBlockwise
  DualScoreBlockwise
  ContractingBlockwise
  OwnedSeparate(Int)
  OwnedContracting(Int)
  MatrixBf16
} derive(Debug, Eq)

///|
pub extend DecodeNumericLaw with Eq::{equal, not_equal}

///|
/// The exact immutable numerical/storage descriptor admitted at startup.
/// It is not a tuning ID, runtime heuristic, or permission to reinterpret an
/// older artifact. KV extent and stage count participate in bundle identity.
pub struct DecodeExecutionDescriptor {
  priv law : DecodeNumericLaw
  priv key_value_tile_rows : Int
  priv pipeline_stages : Int
  priv partition_count : Int
} derive(Debug, Eq)

///|
pub extend DecodeExecutionDescriptor with Eq::{equal, not_equal}

///|
pub fn DecodeExecutionDescriptor::new(
  law~ : DecodeNumericLaw,
  key_value_tile_rows~ : Int,
  pipeline_stages~ : Int,
  partition_count? : Int = 8,
) -> DecodeExecutionDescriptor raise FusedArtifactAdmissionError {
  let legal_law = match law {
    OwnedSeparate(owners) => owners is (4 | 8)
    OwnedContracting(owners) => owners is (2 | 4 | 8)
    _ => true
  }
  if !legal_law ||
    !(key_value_tile_rows is (32 | 64)) ||
    !(pipeline_stages is (1 | 2)) ||
    (law != MatrixBf16 && law != OrderedBlockwise && pipeline_stages != 2) ||
    partition_count < 2 ||
    partition_count > 64 {
    raise InvalidFusedArtifact(KernelAbi)
  }
  { law, key_value_tile_rows, pipeline_stages, partition_count, }
}

///|
pub fn DecodeExecutionDescriptor::from_numeric_law(
  name : String,
  key_value_tile_rows~ : Int,
  pipeline_stages~ : Int,
  partition_count? : Int = 8,
) -> DecodeExecutionDescriptor raise FusedArtifactAdmissionError {
  let law = match name {
    "blockwise-f32-probability-v1" => OrderedBlockwise
    "dual-score-blockwise-f32-probability-v2" => DualScoreBlockwise
    "blockwise-fma-f32-probability-v3" => ContractingBlockwise
    "owned2-blockwise-fma-f32-probability-v4" => OwnedContracting(2)
    "owned4-blockwise-f32-probability-v4" => OwnedSeparate(4)
    "owned4-blockwise-fma-f32-probability-v4" => OwnedContracting(4)
    "owned8-blockwise-f32-probability-v4" => OwnedSeparate(8)
    "owned8-blockwise-fma-f32-probability-v4" => OwnedContracting(8)
    "grouped-head-matrix-bf16-probability-v1" => MatrixBf16
    _ => raise InvalidFusedArtifact(KernelAbi)
  }
  DecodeExecutionDescriptor::new(
    law~,
    key_value_tile_rows~,
    pipeline_stages~,
    partition_count~,
  )
}

///|
/// Complete-chain decomposition is startup data, not a token-step heuristic.
/// Eight is the historical bundle contract, never inferred for a newer bundle.
pub fn DecodeExecutionDescriptor::partition_count(
  self : DecodeExecutionDescriptor,
) -> Int {
  self.partition_count
}

///|
pub fn DecodeExecutionDescriptor::numeric_law(
  self : DecodeExecutionDescriptor,
) -> String {
  match self.law {
    OrderedBlockwise => "blockwise-f32-probability-v1"
    DualScoreBlockwise => "dual-score-blockwise-f32-probability-v2"
    ContractingBlockwise => "blockwise-fma-f32-probability-v3"
    OwnedSeparate(owners) => "owned\{owners}-blockwise-f32-probability-v4"
    OwnedContracting(owners) =>
      "owned\{owners}-blockwise-fma-f32-probability-v4"
    MatrixBf16 => "grouped-head-matrix-bf16-probability-v1"
  }
}

///|
pub fn DecodeExecutionDescriptor::function_symbol(
  self : DecodeExecutionDescriptor,
) -> String {
  let suffix = if self.law == MatrixBf16 {
    "grouped_head_matrix_bf16_v1"
  } else {
    self.numeric_law().replace(old="-probability", new="").split("-").join("_")
  }
  "lunaflux_attention_decode_tile_compiler_v1_" + suffix
}

///|
/// A candidate-qualified AOT symbol may retain its offline compilation tag.
/// The tag grants no numerical or geometry authority: those come only from
/// this descriptor and the signed module/launch fields.
pub fn DecodeExecutionDescriptor::matches_function_symbol(
  self : DecodeExecutionDescriptor,
  symbol : String,
) -> Bool {
  let expected = self.function_symbol()
  if symbol == expected {
    return true
  }
  let prefix = "lunaflux_attention_decode_tile_compiler_v1_"
  guard expected.split(prefix).to_array() is ["", suffix] &&
    symbol.split(prefix).to_array() is ["", tail] &&
    tail.split("_" + suffix.to_string()).to_array() is [tag, ""] &&
    tag.has_prefix("c") &&
    tag.length() >= 2 &&
    tag.length() <= 10 else {
    return false
  }
  let mut index = 0
  for unit in tag {
    if index == 1 && unit == '0' {
      return false
    }
    if index > 0 && !(unit is ('0'..='9')) {
      return false
    }
    index += 1
  }
  true
}

///|
pub fn DecodeExecutionDescriptor::compute(
  self : DecodeExecutionDescriptor,
) -> @luna_attention_strategy.AttentionTileCompute {
  match self.law {
    OrderedBlockwise => AttentionGroupedQueryBlockwiseDot
    DualScoreBlockwise => AttentionGroupedQueryDualScoreBlockwiseDot
    ContractingBlockwise => AttentionGroupedQueryFmaBlockwiseDot
    OwnedSeparate(owners) =>
      AttentionGroupedQueryOwnedBlockwiseDot(
        score_owners=owners,
        contraction=false,
      )
    OwnedContracting(owners) =>
      AttentionGroupedQueryOwnedBlockwiseDot(
        score_owners=owners,
        contraction=true,
      )
    MatrixBf16 => AttentionGroupedQueryMatrixQkPv
  }
}

///|
pub fn DecodeExecutionDescriptor::storage_geometry(
  self : DecodeExecutionDescriptor,
) -> (Int, Int) {
  (self.key_value_tile_rows, self.pipeline_stages)
}

///|
pub fn ReusableFusedRuntimeKernelAbi::is_compiler_blockwise_decode(
  self : ReusableFusedRuntimeKernelAbi,
) -> Bool {
  self
  is (PagedAttentionReadOnlyBlockwiseF32DecodeProductionV8
  | PagedAttentionReadOnlyBlockwiseFmaF32DecodeProductionV9
  | PagedAttentionReadOnlyDualScoreBlockwiseF32DecodeProductionV10
  | PagedAttentionReadOnlyExtendedDecodeProductionV11)
}

///|
fn reusable_runtime_decode_execution(
  abi : ReusableFusedRuntimeKernelAbi,
  supplied : DecodeExecutionDescriptor?,
) -> DecodeExecutionDescriptor? raise FusedArtifactAdmissionError {
  let old_law = match abi {
    PagedAttentionReadOnlyBlockwiseF32DecodeProductionV8 =>
      Some(OrderedBlockwise)
    PagedAttentionReadOnlyBlockwiseFmaF32DecodeProductionV9 =>
      Some(ContractingBlockwise)
    PagedAttentionReadOnlyDualScoreBlockwiseF32DecodeProductionV10 =>
      Some(DualScoreBlockwise)
    _ => None
  }
  match (abi, supplied, old_law) {
    (PagedAttentionReadOnlyExtendedDecodeProductionV11, Some(descriptor), _) =>
      Some(descriptor)
    (PagedAttentionReadOnlyExtendedDecodeProductionV11, None, _) =>
      raise InvalidFusedArtifact(KernelAbi)
    (_, Some(descriptor), Some(law)) => {
      if descriptor.law != law {
        raise InvalidFusedArtifact(KernelAbi)
      }
      Some(descriptor)
    }
    (_, None, Some(law)) =>
      Some(
        DecodeExecutionDescriptor::new(
          law~,
          key_value_tile_rows=32,
          pipeline_stages=2,
        ),
      )
    (_, Some(_), None) => raise InvalidFusedArtifact(KernelAbi)
    _ => None
  }
}

///|
pub fn ApprovedReusableFusedRuntimeModule::decode_execution(
  self : ApprovedReusableFusedRuntimeModule,
) -> DecodeExecutionDescriptor? {
  self.decode_execution
}
