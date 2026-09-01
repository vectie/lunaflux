///|
/// Exact production ABI consumed after a separately admitted QKV projection.
/// The name records the first model-plan consumer without putting model-family
/// branching into the lowering implementation.
pub(all) enum QkNormRopePagedKvWriteKernelAbi {
  QwenQkNormPositionedRopePagedKvWriteProductionV1
} derive(Debug, Eq)

///|
pub fn QkNormRopePagedKvWriteKernelAbi::canonical_name(
  self : QkNormRopePagedKvWriteKernelAbi,
) -> String {
  match self {
    QwenQkNormPositionedRopePagedKvWriteProductionV1 =>
      "step_counts,query_positions,query_row_offsets,sequence_lengths," +
      "page_table_offsets,page_table_page_indices,projected_qkv," +
      "query_norm_weight,key_norm_weight,rotated_qkv,key_cache,value_cache"
  }
}

///|
/// Raw-pointer non-aliasing is a launch precondition, not an inference made by
/// generated CUDA. Operands 0..<9 are read-only; operands 9..<12 are writable
/// and must be disjoint from every other operand and from one another.
pub struct QkNormRopePagedKvWriteAliasContract {
  priv first_writable_operand : Int
  priv operand_count : Int
}

///|
pub fn QkNormRopePagedKvWriteAliasContract::requires_nonoverlap(
  self : QkNormRopePagedKvWriteAliasContract,
  left : Int,
  right : Int,
) -> Bool {
  if left < 0 ||
    right < 0 ||
    left >= self.operand_count ||
    right >= self.operand_count ||
    left == right {
    false
  } else {
    left >= self.first_writable_operand || right >= self.first_writable_operand
  }
}

///|
pub fn QkNormRopePagedKvWriteAliasContract::canonical_name(
  self : QkNormRopePagedKvWriteAliasContract,
) -> String {
  "readonly[0," +
  self.first_writable_operand.to_string() +
  ");writable[" +
  self.first_writable_operand.to_string() +
  "," +
  self.operand_count.to_string() +
  ");writable-disjoint-from-all"
}

///|
pub struct QkNormRopePagedKvWriteAotLimits {
  priv max_source_bytes : Int
  priv max_recipe_bytes : Int
  priv max_module_bytes : Int64
  priv max_receipt_bytes : Int
  priv max_binding_bytes : Int
  priv maximum_grid_x : Int
}

///|
pub fn QkNormRopePagedKvWriteAotLimits::new(
  max_source_bytes~ : Int,
  max_recipe_bytes~ : Int,
  max_module_bytes~ : Int64,
  max_receipt_bytes~ : Int,
  max_binding_bytes~ : Int,
  maximum_grid_x? : Int = 4096,
) -> QkNormRopePagedKvWriteAotLimits raise QkNormRopePagedKvWriteAotError {
  if max_source_bytes <= 0 ||
    max_source_bytes > 1048576 ||
    max_recipe_bytes <= 0 ||
    max_recipe_bytes > 1048576 ||
    max_module_bytes <= 0L ||
    max_module_bytes > 1073741824L ||
    max_receipt_bytes <= 0 ||
    max_receipt_bytes > 1048576 ||
    max_binding_bytes <= 0 ||
    max_binding_bytes > 1048576 ||
    maximum_grid_x <= 0 ||
    maximum_grid_x > 65535 {
    raise Invalid(Limits)
  }
  {
    max_source_bytes,
    max_recipe_bytes,
    max_module_bytes,
    max_receipt_bytes,
    max_binding_bytes,
    maximum_grid_x,
  }
}

///|
pub struct QkNormRopePagedKvWriteCudaCandidate {
  priv semantic : @luna_fusion_plan.QkvQkNormPositionedRopePagedKvWritePlan
  priv model_content : String
  priv model_plan : String
  priv target : @catalog.DeviceTarget
  priv profile : @launch_contract.PagedKernelProfile
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv layout : @device_layout.DeviceKvLayout
  priv operands : ReadOnlyArray[@launch_contract.AotOperand]
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv symbol : String
  priv abi : QkNormRopePagedKvWriteKernelAbi
  priv alias_contract : QkNormRopePagedKvWriteAliasContract
  priv source : Bytes
  priv source_digest : QkNormRopePagedKvWriteDigest
  priv recipe : Bytes
  priv recipe_digest : QkNormRopePagedKvWriteDigest
  priv candidate_digest : QkNormRopePagedKvWriteDigest
}

///|
/// Reproducibly compiled bytes and the exact offline receipt. This has no
/// launch or manifest authority until joined back to its candidate.
pub struct QkNormRopePagedKvWriteCompiledModule {
  priv candidate_digest : QkNormRopePagedKvWriteDigest
  priv module_bytes : Bytes
  priv module_digest : QkNormRopePagedKvWriteDigest
  priv receipt : Bytes
  priv receipt_digest : QkNormRopePagedKvWriteDigest
  priv driver_identity : String
}

///|
/// Exact candidate-to-module join. Physical qualification and external
/// approval remain separate, so this binding is deliberately non-promoting.
pub struct QkNormRopePagedKvWriteCudaBinding {
  priv candidate : QkNormRopePagedKvWriteCudaCandidate
  priv compiled : QkNormRopePagedKvWriteCompiledModule
  priv canonical : Bytes
  priv digest : QkNormRopePagedKvWriteDigest
}
