///|
pub(all) enum JointAudioAmpParameterRole {
  ActivationAlpha(Int32, Int32)
  ActivationBeta(Int32, Int32)
  ActivationDownsampleFilter(Int32, Int32)
  ActivationUpsampleFilter(Int32, Int32)
  ConvolutionBias(Int32, Int32, Int32)
  ConvolutionWeightG(Int32, Int32, Int32)
  ConvolutionWeightV(Int32, Int32, Int32)
} derive(Debug, Eq)

///|
pub(all) suberror JointAudioAmpParameterLayoutError {
  InvalidRegionCount
  InvalidRegion(Int32)
  InvalidRegionOrder(Int32)
  InvalidRegionOffset(Int32)
  DuplicateRegion(Int32)
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
