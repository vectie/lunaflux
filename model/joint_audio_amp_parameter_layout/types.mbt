///|
pub(all) enum JointAudioAmpParameterRole {
  ActivationAlpha(Int, Int)
  ActivationBeta(Int, Int)
  ActivationDownsampleFilter(Int, Int)
  ActivationUpsampleFilter(Int, Int)
  ConvolutionBias(Int, Int, Int)
  ConvolutionWeightG(Int, Int, Int)
  ConvolutionWeightV(Int, Int, Int)
} derive(Debug, Eq)

///|
pub(all) suberror JointAudioAmpParameterLayoutError {
  InvalidRegionCount
  InvalidRegion(Int)
  InvalidRegionOrder(Int)
  InvalidRegionOffset(Int)
  DuplicateRegion(Int)
  SizeOverflow
} derive(Debug, Eq)

///|
pub struct JointAudioAmpParameterRegionInput {
  priv role : JointAudioAmpParameterRole
  priv tensor_name : String
  priv element_count : Int64
  priv byte_offset : Int64
}

///|
pub struct JointAudioAmpParameterRegion {
  priv role : JointAudioAmpParameterRole
  priv tensor_name : String
  priv element_count : Int64
  priv byte_offset : Int64
  priv byte_count : Int64
} derive(Debug, Eq)

///|
pub struct JointAudioAmpParameterLayoutDigest {
  priv hex : String
} derive(Debug, Eq)

///|
pub struct JointAudioAmpParameterLayout {
  priv regions : ReadOnlyArray[JointAudioAmpParameterRegion]
  priv total_arena_bytes : Int64
  priv digest : JointAudioAmpParameterLayoutDigest
}
