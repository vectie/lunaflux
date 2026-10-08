///|
/// Named prefill numerical permissions, separate from launch geometry.
/// ApproximateBaseTwoExponential changes only the elementary exponential;
/// ReferenceBf16ProbabilityBaseTwo names a separate algorithm and probability
/// precision. Neither grants arbitrary dtype conversion or accuracy approval.
pub(all) enum PrefillExponentialLaw {
  StrictNaturalExponential
  ApproximateBaseTwoExponential
  /// Separate algorithm/numerical contract, not an expf peephole rewrite.
  ReferenceBf16ProbabilityBaseTwo
} derive(Debug, Eq)

///|
pub fn PrefillExponentialLaw::numeric_law(
  self : PrefillExponentialLaw,
) -> String {
  match self {
    StrictNaturalExponential => "strict-natural-exponential-v1"
    ApproximateBaseTwoExponential => "approx-base2-f32-v1"
    ReferenceBf16ProbabilityBaseTwo =>
      "reference-flashattention-bf16-probability-exp2-v1"
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
      if symbol.contains("_exp2_v1") || symbol.contains("_flashattention_") {
        raise InvalidFusedArtifact(KernelAbi)
      }
    ApproximateBaseTwoExponential =>
      if !allow_approximate ||
        !reusable_runtime_has_prefill_exponential(abi) ||
        !symbol.contains("_exp2_v1") ||
        symbol.contains("_flashattention_") {
        raise InvalidFusedArtifact(KernelAbi)
      }
    ReferenceBf16ProbabilityBaseTwo =>
      if !allow_approximate ||
        !(abi
        is (QwenPagedAttentionReadOnlyProductionV2
        | QwenPagedAttentionPrefillWideQueryProductionV5)) ||
        !(symbol
        is ("lunaflux_attention_prefill_flashattention_bf16_exp2_v1"
        | "lunaflux_attention_mixed_flashattention_bf16_exp2_v1")) {
        raise InvalidFusedArtifact(KernelAbi)
      }
  }
}

///|
/// Decode the immutable ABI's output ownership at startup. Runtime planning
/// consumes this semantic domain, not a CUDA symbol or model-name heuristic.
pub fn ApprovedReusableFusedRuntimeModule::mixed_row_coverage(
  self : ApprovedReusableFusedRuntimeModule,
) -> @luna_attention_strategy.AttentionMixedRowCoverage {
  if self.prefill_exponential_law == ReferenceBf16ProbabilityBaseTwo &&
    self.symbol == "lunaflux_attention_mixed_flashattention_bf16_exp2_v1" {
    AllActiveRows
  } else {
    PrefillRowsOnly
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
      "reference-flashattention-bf16-probability-exp2-v1" =>
        ReferenceBf16ProbabilityBaseTwo
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
