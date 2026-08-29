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
  priv id : @plan.OperationId
  priv kind : @plan.OperationKind
  priv shape : @plan.OperationShape
  priv inputs : ReadOnlyArray[StaticInput]
  priv outputs : ReadOnlyArray[@plan.ActivationRef]
  priv binding : @catalog.KernelBinding
  priv operation_execution_digest : @numeric_contract.OperationExecutionDigest
} derive(Debug, Eq)

///|
/// Immutable startup bridge from semantic model data to device execution.
///
/// This value owns no device resource and launches no work. It pins exact
/// model, weight-arena, workspace, target, and kernel-selection facts for a
/// later executor.
pub struct StaticDevicePlan {
  priv identity : @spec.ModelIdentity
  priv shape_constraints : @plan.ShapeConstraints
  priv target : @catalog.DeviceTarget
  priv catalog_version : @catalog.CatalogVersion
  priv numeric_schema_digest : @numeric_contract.ModelNumericSchemaDigest
  priv weight_layout : @device_materialize.DeviceWeightLayout
  priv workspace_bytes : Int64
  priv workspace_alignment : Int64
  priv operations : ReadOnlyArray[StaticOperation]
  priv final_output : @plan.ActivationRef
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
pub fn StaticOperation::operation_execution_digest(
  self : StaticOperation,
) -> @numeric_contract.OperationExecutionDigest {
  self.operation_execution_digest
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
pub fn StaticDevicePlan::numeric_schema_digest(
  self : StaticDevicePlan,
) -> @numeric_contract.ModelNumericSchemaDigest {
  self.numeric_schema_digest
}

///|
/// Returns the complete already-authenticated parameter and numeric-metadata
/// placement. This is inert read-only planning evidence, not an allocation.
pub fn StaticDevicePlan::weight_layout(
  self : StaticDevicePlan,
) -> @device_materialize.DeviceWeightLayout {
  self.weight_layout
}

///|
pub fn StaticDevicePlan::weight_arena_bytes(self : StaticDevicePlan) -> Int64 {
  self.weight_layout.arena_bytes()
}

///|
pub fn StaticDevicePlan::weight_alignment(self : StaticDevicePlan) -> Int64 {
  self.weight_layout.alignment_bytes()
}

///|
pub fn StaticDevicePlan::weight_tensor_count(self : StaticDevicePlan) -> Int {
  self.weight_layout.length()
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
