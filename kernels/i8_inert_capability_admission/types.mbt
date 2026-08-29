///|
/// Version of the catalog-only symmetric-I8 admission record.
pub struct I8InertKernelCapabilityAdmissionVersion {
  priv value : Int
} derive(Debug, Eq)

///|
pub fn I8InertKernelCapabilityAdmissionVersion::v1() -> I8InertKernelCapabilityAdmissionVersion {
  { value: 1 }
}

///|
pub fn I8InertKernelCapabilityAdmissionVersion::as_int(
  self : I8InertKernelCapabilityAdmissionVersion,
) -> Int {
  self.value
}

///|
/// SHA-256 identity of one complete catalog-only inert admission.
pub struct I8InertKernelCapabilityAdmissionDigest {
  priv value : String
} derive(Debug, Eq, Hash)

///|
pub fn I8InertKernelCapabilityAdmissionDigest::as_hex(
  self : I8InertKernelCapabilityAdmissionDigest,
) -> String {
  self.value
}

///|
/// Startup bounds for derived evidence. No request-path value reaches these
/// limits and no resource is acquired by their construction.
pub struct I8InertKernelCapabilityAdmissionLimits {
  priv max_operations : Int
  priv max_weights : Int
  priv max_canonical_bytes : Int
} derive(Debug, Eq)

///|
pub fn I8InertKernelCapabilityAdmissionLimits::new(
  max_operations~ : Int,
  max_weights~ : Int,
  max_canonical_bytes~ : Int,
) -> I8InertKernelCapabilityAdmissionLimits raise I8InertKernelCapabilityAdmissionError {
  if max_operations <= 0 ||
    max_operations > I8_ADMISSION_MAX_OPERATIONS ||
    max_weights <= 0 ||
    max_weights > I8_ADMISSION_MAX_WEIGHTS ||
    max_weights < max_operations ||
    max_canonical_bytes <= 0 ||
    max_canonical_bytes > I8_ADMISSION_MAX_CANONICAL_BYTES ||
    i8_admission_v1_size_within(
      max_operations, max_weights, max_canonical_bytes,
    )
    is None {
    raise I8InertKernelCapabilityRejected(Limit)
  }
  { max_operations, max_weights, max_canonical_bytes }
}

///|
pub(all) enum I8InertKernelCapabilityAdmissionRule {
  Limit
  ExecutionScope
  ModelIdentity
  NumericSchema
  DeviceCapability
  DeviceTarget
  CatalogVersion
  BindingCount
  BindingOrder
  OperationExecution
  UnsupportedI8Operation
  MissingI8Operation
  MixedWeightStorage
  ScaleBinding
  DuplicateScale
  AotBinding
  Workspace
  CanonicalBytes
  ProductionTensorStorage
  ProductionExecution
  ProductionScaleCoverage
  ProductionGraph
  MixedFp8Execution
} derive(Debug, Eq)

///|
pub(all) suberror I8InertKernelCapabilityAdmissionError {
  I8InertKernelCapabilityRejected(I8InertKernelCapabilityAdmissionRule)
} derive(Debug, Eq)

///|
/// Exact inert schema evidence for one selected I8 weight and its distinct
/// per-output-channel F32 scale. These are identities and shapes, not storage.
pub struct I8InertWeightEvidence {
  priv tensor : @plan.TensorRef
  priv storage : @numeric_contract.TensorStorageDigest
  priv shape : @numeric_contract.NumericShape
  priv scale : @plan.ScaleTensorRef
  priv scale_storage : @numeric_contract.TensorStorageDigest
  priv scale_shape : @numeric_contract.NumericShape
  priv output_channel_count : Int64
} derive(Debug, Eq)

///|
/// Exact catalog identity for one selected symmetric-I8 operation.
/// The entry point is a content-addressed value only; it is not a loaded
/// module, symbol, launch contract, or callable handle.
pub struct I8InertOperationEvidence {
  priv operation : @plan.OperationId
  priv operation_kind : @plan.OperationKind
  priv capability : @plan.KernelCapabilityId
  priv execution : @numeric_contract.OperationExecutionDigest
  priv semantic_version : @catalog.KernelSemanticVersion
  priv entry_point : @catalog.AotKernelEntryPoint
  priv workspace_bytes : Int64
  priv workspace_alignment : Int64
  priv weights : ReadOnlyArray[I8InertWeightEvidence]
} derive(Debug, Eq)

///|
/// One pure catalog-only admission for symmetric-I8 weight-only execution.
///
/// This value owns no model plan, resolved catalog, artifact bytes, module,
/// symbol, launch contract, device plan, executor, or readiness authority.
pub struct I8InertKernelCapabilityAdmission {
  priv version : I8InertKernelCapabilityAdmissionVersion
  priv model_identity : @spec.ModelIdentity
  priv numeric_schema : @numeric_contract.ModelNumericSchemaDigest
  priv device_numeric : @device.DeviceNumericCapabilityDigest
  priv target : @catalog.DeviceTarget
  priv catalog_version : @catalog.CatalogVersion
  priv operations : ReadOnlyArray[I8InertOperationEvidence]
  priv canonical : Bytes
  priv digest : I8InertKernelCapabilityAdmissionDigest
} derive(Debug, Eq)

///|
/// Exact production policy layered over the catalog-only admission.
///
/// This record proves that the complete paged model schema contains only
/// plain BF16 parameters and reviewed symmetric-I8 weight-only parameters,
/// that every I8 parameter owns one unique F32 per-output-channel scale, and
/// that no FP8 execution or metadata is mixed into the graph. It remains
/// inert: no artifact, launch, loader, device, or executor authority is held.
pub struct I8ProductionPolicyAdmission {
  priv admission : I8InertKernelCapabilityAdmission
  priv i8_parameter_count : Int
  priv scale_tensor_count : Int
} derive(Debug, Eq)
