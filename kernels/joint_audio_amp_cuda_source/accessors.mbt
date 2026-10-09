///|
pub fn JointAudioAmpCudaSource::numerical_contract(
  self : Self,
) -> JointAudioAmpCudaNumericalContract {
  self.numerical_contract
}

///|
pub fn JointAudioAmpCudaSource::element_count(self : Self) -> Int64 {
  self.element_count
}

///|
pub fn JointAudioAmpCudaSource::residual_symbol(self : Self) -> String {
  self.residual_symbol
}

///|
pub fn JointAudioAmpCudaSource::average_symbol(self : Self) -> String {
  self.average_symbol
}

///|
pub fn JointAudioAmpCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}
