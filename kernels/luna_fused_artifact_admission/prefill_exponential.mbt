///|
/// Elementary exponential permission, independent of tile or launch geometry.
/// The approximate law does not grant reassociation, contraction or a different
/// probability/storage dtype. It requires a separately qualified artifact.
pub(all) enum PrefillExponentialLaw {
  StrictNaturalExponential
  ApproximateBaseTwoExponential
} derive(Debug, Eq)

///|
pub fn PrefillExponentialLaw::numeric_law(
  self : PrefillExponentialLaw,
) -> String {
  match self {
    StrictNaturalExponential => "strict-natural-exponential-v1"
    ApproximateBaseTwoExponential => "approx-base2-f32-v1"
  }
}

///|
fn reusable_runtime_has_prefill_exponential(
  abi : ReusableFusedRuntimeKernelAbi,
) -> Bool {
  abi
  is (QwenPagedAttentionReadOnlyProductionV2
  | QwenPagedAttentionPrefillWideQueryProductionV5
  | QwenPagedAttentionPrefillPartitionedPartialProductionV4
  | QwenPagedAttentionPrefillPartitioned8PartialProductionV6)
}

///|
fn reusable_runtime_validate_prefill_exponential(
  abi : ReusableFusedRuntimeKernelAbi,
  law : PrefillExponentialLaw,
  symbol : String,
  allow_approximate : Bool,
) -> Unit raise FusedArtifactAdmissionError {
  match law {
    StrictNaturalExponential =>
      if symbol.contains("_exp2_v1") {
        raise InvalidFusedArtifact(KernelAbi)
      }
    ApproximateBaseTwoExponential =>
      if !allow_approximate ||
        !reusable_runtime_has_prefill_exponential(abi) ||
        !symbol.contains("_exp2_v1") {
        raise InvalidFusedArtifact(KernelAbi)
      }
  }
}

///|
fn reusable_runtime_append_prefill_exponential(
  out : StringBuilder,
  prefix : String,
  law : PrefillExponentialLaw,
) -> Unit {
  out.write_string(
    prefix + "prefill_exponential_law=" + law.numeric_law() + "\n",
  )
}

///|
fn reusable_runtime_read_prefill_exponential(
  reader : FusedDeploymentReader,
  prefix : String,
  abi : ReusableFusedRuntimeKernelAbi,
  symbol : String,
  has_numeric : Bool,
  allow_approximate : Bool,
) -> PrefillExponentialLaw raise FusedArtifactAdmissionError {
  let law = if has_numeric {
    match reader.value(prefix + "prefill_exponential_law") {
      "strict-natural-exponential-v1" => StrictNaturalExponential
      "approx-base2-f32-v1" => ApproximateBaseTwoExponential
      _ => raise InvalidFusedArtifact(KernelAbi)
    }
  } else {
    StrictNaturalExponential
  }
  reusable_runtime_validate_prefill_exponential(
    abi, law, symbol, allow_approximate,
  )
  law
}

///|
pub fn ApprovedReusableFusedRuntimeModule::prefill_exponential_law(
  self : ApprovedReusableFusedRuntimeModule,
) -> PrefillExponentialLaw {
  self.prefill_exponential_law
}
