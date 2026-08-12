///|
/// One fully resolved input to a static semantic operation.
///
/// Weight inputs carry checked byte regions, while activation and model-input
/// references remain semantic identifiers. No allocation or raw address is
/// stored in this plan.
pub(all) enum StaticInput {
  ExternalInput(@plan.ModelInputRef)
  WeightRegion(@device_materialize.DeviceWeightRegion)
  ActivationValue(@plan.ActivationRef)
} derive(Debug, Eq)

///|
/// One immutable semantic operation paired with its exact startup-selected
/// kernel binding.
pub struct StaticOperation {
  id : @plan.OperationId
  kind : @plan.OperationKind
  shape : @plan.OperationShape
  inputs : ReadOnlyArray[StaticInput]
  outputs : ReadOnlyArray[@plan.ActivationRef]
  binding : @catalog.KernelBinding
} derive(Debug, Eq)

///|
/// Immutable startup bridge from semantic model data to device execution.
///
/// This value owns no device resource and launches no work. It pins exact
/// model, weight-arena, workspace, target, and kernel-selection facts for a
/// later executor.
pub struct StaticDevicePlan {
  identity : @spec.ModelIdentity
  shape_constraints : @plan.ShapeConstraints
  target : @catalog.DeviceTarget
  catalog_version : @catalog.CatalogVersion
  weight_arena_bytes : Int64
  weight_alignment : Int64
  weight_tensor_count : Int
  workspace_bytes : Int64
  workspace_alignment : Int64
  operations : ReadOnlyArray[StaticOperation]
  final_output : @plan.ActivationRef
} derive(Debug, Eq)

///|
pub fn StaticOperation::id(self : StaticOperation) -> @plan.OperationId {
  self.id
}

///|
pub fn StaticOperation::kind(self : StaticOperation) -> @plan.OperationKind {
  self.kind
}

///|
pub fn StaticOperation::shape(self : StaticOperation) -> @plan.OperationShape {
  self.shape
}

///|
pub fn StaticOperation::inputs(
  self : StaticOperation,
) -> ReadOnlyArray[StaticInput] {
  self.inputs
}

///|
pub fn StaticOperation::outputs(
  self : StaticOperation,
) -> ReadOnlyArray[@plan.ActivationRef] {
  self.outputs
}

///|
pub fn StaticOperation::binding(
  self : StaticOperation,
) -> @catalog.KernelBinding {
  self.binding
}

///|
pub fn StaticDevicePlan::identity(
  self : StaticDevicePlan,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn StaticDevicePlan::shape_constraints(
  self : StaticDevicePlan,
) -> @plan.ShapeConstraints {
  self.shape_constraints
}

///|
pub fn StaticDevicePlan::target(
  self : StaticDevicePlan,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn StaticDevicePlan::catalog_version(
  self : StaticDevicePlan,
) -> @catalog.CatalogVersion {
  self.catalog_version
}

///|
pub fn StaticDevicePlan::weight_arena_bytes(self : StaticDevicePlan) -> Int64 {
  self.weight_arena_bytes
}

///|
pub fn StaticDevicePlan::weight_alignment(self : StaticDevicePlan) -> Int64 {
  self.weight_alignment
}

///|
pub fn StaticDevicePlan::weight_tensor_count(self : StaticDevicePlan) -> Int {
  self.weight_tensor_count
}

///|
pub fn StaticDevicePlan::workspace_bytes(self : StaticDevicePlan) -> Int64 {
  self.workspace_bytes
}

///|
pub fn StaticDevicePlan::workspace_alignment(self : StaticDevicePlan) -> Int64 {
  self.workspace_alignment
}

///|
pub fn StaticDevicePlan::operations(
  self : StaticDevicePlan,
) -> ReadOnlyArray[StaticOperation] {
  self.operations
}

///|
pub fn StaticDevicePlan::final_output(
  self : StaticDevicePlan,
) -> @plan.ActivationRef {
  self.final_output
}
