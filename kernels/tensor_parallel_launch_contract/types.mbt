///|
/// Untrusted bounded artifact/profile input for one exact operation.
///
/// Construction grants no launch, module, or device authority. Admission
/// independently derives and compares every repeated shard scalar before the
/// input can appear in an opaque admitted contract.
pub struct TensorParallelAotLaunchContractInput {
  priv profile_id : @launch_contract.KernelProfileId
  priv operation_id : @plan.OperationId
  priv entry_point : @catalog.AotKernelEntryPoint
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv rank : Int32
  priv world_size : Int32
  priv device_target : @catalog.DeviceTarget
  priv max_query_rows : Int32
  priv max_query_tokens : Int32
  priv max_page_table_entries : Int32
  priv input_width : Int32
  priv output_width : Int32
  priv inner_width : Int32
  priv query_head_count : Int32
  priv kv_head_count : Int32
  priv head_dimension : Int32
  priv kv_layout_version : Int32
  priv tokens_per_page : Int32
} derive(Debug, Eq)

///|
pub(all) enum TensorParallelOperandRole {
  RuntimeInput(@plan.OperationRuntimeInput)
  RankConstants(version~ : Int32)
  ExternalInput(@plan.ModelInputRef)
  WeightInput(@plan.TensorRef)
  ActivationInput(@plan.ActivationRef)
  ActivationOutput(@plan.ActivationRef)
  PersistentKvComponent(@plan.KvLayerId, @device_layout.DeviceKvComponent)
  Workspace
} derive(Debug, Eq)

///|
pub struct TensorParallelOperand {
  priv role : TensorParallelOperandRole
  priv byte_count : Int64
  priv alignment : Int64
} derive(Debug, Eq)

///|
/// Exact scalar ABI uploaded once and shared by every admitted rank launch.
pub struct TensorParallelRankConstants {
  priv version : Int32
  priv byte_count : Int64
  priv alignment : Int64
  priv rank : Int32
  priv world_size : Int32
  priv compute_major : Int32
  priv compute_minor : Int32
  priv kv_layout_version : Int32
  priv tokens_per_page : Int32
} derive(Debug, Eq)

///|
pub(all) enum TensorParallelCollectiveAfter {
  NoCollective
  CollectiveSumAllReduce(sequence~ : Int32, width~ : Int32)
  CollectiveLastAxisAllGather(
    sequence~ : Int32,
    full_width~ : Int32,
    local_width~ : Int32
  )
} derive(Debug, Eq)

///|
pub struct TensorParallelAotLaunchContract {
  priv profile : @launch_contract.PagedKernelProfile
  priv operation_id : @plan.OperationId
  priv operation_kind : @plan.OperationKind
  priv capability : @plan.KernelCapabilityId
  priv entry_point : @catalog.AotKernelEntryPoint
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv input_width : Int32
  priv output_width : Int32
  priv inner_width : Int32
  priv query_head_count : Int32
  priv kv_head_count : Int32
  priv head_dimension : Int32
  priv operands : ReadOnlyArray[TensorParallelOperand]
  priv collective_after : TensorParallelCollectiveAfter
} derive(Debug, Eq)

///|
pub struct TensorParallelAotLaunchContractSet {
  priv model_identity : @spec.ModelIdentity
  priv rank : Int32
  priv world_size : Int32
  priv device_target : @catalog.DeviceTarget
  priv catalog_version : @catalog.CatalogVersion
  priv device_kv_layout : @device_layout.DeviceKvLayout
  priv rank_constants : TensorParallelRankConstants
  priv profiles : ReadOnlyArray[@launch_contract.PagedKernelProfile]
  priv contracts : ReadOnlyArray[TensorParallelAotLaunchContract]
} derive(Debug, Eq)

///|
pub fn TensorParallelAotLaunchContractInput::new(
  profile_id~ : @launch_contract.KernelProfileId,
  operation_id~ : @plan.OperationId,
  entry_point~ : @catalog.AotKernelEntryPoint,
  dimensions~ : @launch_contract.AotLaunchDimensions,
  rank~ : Int32,
  world_size~ : Int32,
  device_target~ : @catalog.DeviceTarget,
  max_query_rows~ : Int32,
  max_query_tokens~ : Int32,
  max_page_table_entries~ : Int32,
  input_width~ : Int32,
  output_width~ : Int32,
  inner_width~ : Int32,
  query_head_count~ : Int32,
  kv_head_count~ : Int32,
  head_dimension~ : Int32,
  kv_layout_version~ : Int32,
  tokens_per_page~ : Int32,
) -> TensorParallelAotLaunchContractInput {
  {
    profile_id,
    operation_id,
    entry_point,
    dimensions,
    rank,
    world_size,
    device_target,
    max_query_rows,
    max_query_tokens,
    max_page_table_entries,
    input_width,
    output_width,
    inner_width,
    query_head_count,
    kv_head_count,
    head_dimension,
    kv_layout_version,
    tokens_per_page,
  }
}
