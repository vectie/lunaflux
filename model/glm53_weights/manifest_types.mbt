///|
/// Scalar storage expected in the decoded safetensors header. Float32 is
/// deliberately distinct from BF16 routing compute policy.
pub(all) enum Glm53TensorDType {
  BFloat16
  Float32
  Fp8E4M3
} derive(Debug, Eq)

///|
/// One canonical tensor slot in the exact official checkpoint namespace.
pub struct Glm53TensorDescriptor {
  priv ordinal : Int
  priv name : String
  priv dtype : Glm53TensorDType
  priv shape : ReadOnlyArray[Int64]
} derive(Debug, Eq)

///|
/// Exact ordered GLM tensor namespace, bound to authenticated content and the
/// model-family execution-plan identity.
pub struct Glm53TensorManifest {
  priv profile : @glm53_spec.Glm53ProfileId
  priv encoding : Glm53WeightEncoding
  priv execution_identity : @spec.ModelIdentity
  priv layout_digest : @spec.PlanDigest
  priv tensors : ReadOnlyArray[Glm53TensorDescriptor]
  priv control_token_ids : ReadOnlyArray[Int]
} derive(Debug, Eq)

///|
/// Family-owned decoded safetensors header entry. This preserves F32 and FP8
/// metadata that the v1 BF16-only shared materializer intentionally rejects.
pub struct Glm53ObservedTensor {
  priv name : String
  priv dtype : Glm53TensorDType
  priv shape : ReadOnlyArray[Int64]
} derive(Debug, Eq)

///|
/// Immutable startup binding. It grants no payload/materialization authority.
pub struct Glm53WeightBinding {
  priv execution_identity : @spec.ModelIdentity
  priv layout_digest : @spec.PlanDigest
  priv tensor_count : Int
} derive(Debug, Eq)

///|
pub fn Glm53ObservedTensor::new(
  name~ : String,
  dtype~ : Glm53TensorDType,
  shape~ : ArrayView[Int64],
) -> Glm53ObservedTensor {
  { name, dtype, shape: ReadOnlyArray::from_array(shape), }
}

///|
pub fn Glm53TensorDescriptor::ordinal(self : Self) -> Int {
  self.ordinal
}

///|
pub fn Glm53TensorDescriptor::name(self : Self) -> String {
  self.name
}

///|
pub fn Glm53TensorDescriptor::dtype(self : Self) -> Glm53TensorDType {
  self.dtype
}

///|
pub fn Glm53TensorDescriptor::shape(self : Self) -> ReadOnlyArray[Int64] {
  self.shape
}

///|
pub fn Glm53TensorManifest::profile(self : Self) -> @glm53_spec.Glm53ProfileId {
  self.profile
}

///|
pub fn Glm53TensorManifest::encoding(self : Self) -> Glm53WeightEncoding {
  self.encoding
}

///|
pub fn Glm53TensorManifest::execution_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.execution_identity
}

///|
pub fn Glm53TensorManifest::layout_digest(self : Self) -> @spec.PlanDigest {
  self.layout_digest
}

///|
pub fn Glm53TensorManifest::tensor_count(self : Self) -> Int {
  self.tensors.length()
}

///|
pub fn Glm53TensorManifest::tensor(
  self : Self,
  ordinal : Int,
) -> Glm53TensorDescriptor? {
  self.tensors.get(ordinal)
}

///|
pub fn Glm53TensorManifest::control_token_ids(
  self : Self,
) -> ReadOnlyArray[Int] {
  self.control_token_ids
}

///|
pub fn Glm53WeightBinding::execution_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.execution_identity
}

///|
pub fn Glm53WeightBinding::layout_digest(self : Self) -> @spec.PlanDigest {
  self.layout_digest
}

///|
pub fn Glm53WeightBinding::tensor_count(self : Self) -> Int {
  self.tensor_count
}
