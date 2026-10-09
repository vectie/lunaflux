///|
pub fn Glm53NumericTensorBinding::ordinal(self : Self) -> Int {
  self.ordinal
}

///|
pub fn Glm53NumericTensorBinding::role(self : Self) -> Glm53LogicalTensorRole {
  self.role
}

///|
pub fn Glm53NumericOperationBinding::ordinal(self : Self) -> Int {
  self.ordinal
}

///|
pub fn Glm53NumericOperationBinding::scope(
  self : Self,
) -> Glm53NumericOperationScope {
  self.scope
}

///|
pub fn Glm53NumericOperationBinding::kind(
  self : Self,
) -> Glm53NumericOperationKind {
  self.kind
}

///|
pub fn Glm53NumericOperationBinding::execution(
  self : Self,
) -> @numeric_contract.OperationExecutionContract {
  self.execution
}

///|
pub fn Glm53NumericPlan::profile(self : Self) -> @glm53_spec.Glm53ProfileId {
  self.profile
}

///|
pub fn Glm53NumericPlan::encoding(
  self : Self,
) -> @glm53_weights.Glm53WeightEncoding {
  self.encoding
}

///|
pub fn Glm53NumericPlan::execution_identity(self : Self) -> @spec.ModelIdentity {
  self.execution_identity
}

///|
pub fn Glm53NumericPlan::manifest_layout_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.manifest_layout_digest
}

///|
pub fn Glm53NumericPlan::numeric_schema_digest(
  self : Self,
) -> @numeric_contract.ModelNumericSchemaDigest {
  self.numeric_schema.digest()
}

///|
pub fn Glm53NumericPlan::binding_digest(self : Self) -> @spec.PlanDigest {
  self.binding_digest
}

///|
pub fn Glm53NumericPlan::tensor_count(self : Self) -> Int {
  self.tensor_bindings.length()
}

///|
pub fn Glm53NumericPlan::operation_count(self : Self) -> Int {
  self.operation_bindings.length()
}

///|
pub fn Glm53NumericPlan::bfloat16_tensor_count(self : Self) -> Int {
  self.bfloat16_tensor_count
}

///|
pub fn Glm53NumericPlan::float32_tensor_count(self : Self) -> Int {
  self.float32_tensor_count
}

///|
pub fn Glm53NumericPlan::tensor(
  self : Self,
  ordinal : Int,
) -> Glm53NumericTensorBinding? {
  self.tensor_bindings.get(ordinal)
}

///|
pub fn Glm53NumericPlan::operation(
  self : Self,
  ordinal : Int,
) -> Glm53NumericOperationBinding? {
  self.operation_bindings.get(ordinal)
}

///|
pub fn Glm53NumericPlan::numeric_schema(
  self : Self,
) -> @numeric_contract.ModelNumericSchema {
  self.numeric_schema
}

///|
/// Physical admission remains separate from logical numeric planning. Both
/// admitted profiles mix BF16 and F32 logical tensors and therefore fail
/// closed until a mixed-dtype materializer owns an exact contract.
pub fn Glm53NumericPlan::require_physical_materialization(
  self : Self,
) -> Unit raise Glm53NumericPlanError {
  ignore(self)
  raise MixedDTypeMaterializationUnsupported
}
