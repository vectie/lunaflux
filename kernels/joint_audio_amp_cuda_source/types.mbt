///|
pub(all) enum JointAudioAmpCudaNumericalContract {
  OrderedF32ResidualAndThreeWayAverageV1
} derive(Debug, Eq)

///|
pub(all) suberror JointAudioAmpCudaSourceError {
  InvalidElementCount
  InvalidFunctionSymbol
  InvalidReferenceShape
} derive(Debug, Eq)

///|
pub struct JointAudioAmpCudaSource {
  priv numerical_contract : JointAudioAmpCudaNumericalContract
  priv element_count : Int64
  priv residual_symbol : String
  priv average_symbol : String
  priv source : Bytes
}
