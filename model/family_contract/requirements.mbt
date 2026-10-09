///|
pub fn ModelFamilyRequirements::new(
  family~ : ModelFamilyId,
  architecture~ : String,
  model_type~ : String,
  schema_version~ : Int,
  workload~ : ModelWorkload,
  capabilities~ : ArrayView[ModelSemanticCapability],
) -> ModelFamilyRequirements raise FamilyContractError {
  if architecture.is_empty() {
    raise EmptyArchitecture
  }
  if model_type.is_empty() {
    raise EmptyModelType
  }
  if schema_version <= 0 {
    raise InvalidSchemaVersion
  }
  let seen = FixedArray::make(capability_count(), false)
  for capability in capabilities {
    let index = capability.stable_index()
    if seen[index] {
      raise DuplicateCapability(capability)
    }
    seen[index] = true
  }
  {
    family,
    architecture,
    model_type,
    schema_version,
    workload,
    capabilities: ReadOnlyArray::from_array(capabilities),
  }
}

///|
pub fn ModelFamilyRequirements::family(self : Self) -> ModelFamilyId {
  self.family
}

///|
pub fn ModelFamilyRequirements::architecture(self : Self) -> String {
  self.architecture
}

///|
pub fn ModelFamilyRequirements::model_type(self : Self) -> String {
  self.model_type
}

///|
pub fn ModelFamilyRequirements::schema_version(self : Self) -> Int {
  self.schema_version
}

///|
pub fn ModelFamilyRequirements::workload(self : Self) -> ModelWorkload {
  self.workload
}

///|
pub fn ModelFamilyRequirements::capabilities(
  self : Self,
) -> ReadOnlyArray[ModelSemanticCapability] {
  self.capabilities
}

///|
/// Joins semantic capability names only. A successful result is not evidence
/// of weight, artifact, device, runtime, or physical execution readiness.
pub fn ModelFamilyRequirements::admit(
  self : Self,
  backend : ModelBackendCapabilities,
) -> ModelFamilyAdmission {
  let missing : Array[ModelSemanticCapability] = []
  for capability in self.capabilities {
    if !backend.supports(capability) {
      missing.push(capability)
    }
  }
  if missing.is_empty() {
    SemanticCapabilitiesAvailable(self)
  } else {
    MissingCapabilities(self, ReadOnlyArray::from_array(missing))
  }
}
