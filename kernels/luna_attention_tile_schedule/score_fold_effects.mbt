///|
/// Shared-score fold operations. Publication separates cross-owner writes and
/// reads; a resident output fold does not need a shared rescale operation.
pub(all) enum AttentionScoreFoldStep {
  UpdateResidentStatisticsAndProbabilities
  UpdateSharedMaximumAndScale
  RescaleSharedOutput
  UpdateSharedProbabilitiesAndDenominator
  PublishScoreFold
} derive(Debug, Eq)

///|
/// Order the online fold without reassociating its arithmetic. Resident
/// statistics combine the maximum and probability passes in one owner;
/// shared statistics must publish the new scale before other owners use it.
pub fn plan_attention_score_fold(
  resident_statistics~ : Bool,
  resident_output~ : Bool,
) -> ReadOnlyArray[AttentionScoreFoldStep] {
  let prefix : Array[AttentionScoreFoldStep] = if resident_statistics {
    [UpdateResidentStatisticsAndProbabilities, PublishScoreFold]
  } else {
    [UpdateSharedMaximumAndScale, PublishScoreFold]
  }
  if !resident_output {
    prefix.push(RescaleSharedOutput)
    prefix.push(PublishScoreFold)
  }
  if !resident_statistics {
    prefix.push(UpdateSharedProbabilitiesAndDenominator)
    prefix.push(PublishScoreFold)
  }
  ReadOnlyArray::from_array(prefix)
}
