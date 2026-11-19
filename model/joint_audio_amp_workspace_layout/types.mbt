///|
pub(all) enum JointAudioAmpActivationPhase {
  Upsample
  SnakeBeta
  Downsample
} derive(Debug, Eq)

///|
pub(all) enum JointAudioAmpConvolutionPhase {
  Normalize
  Convolve
} derive(Debug, Eq)

///|
pub(all) enum JointAudioAmpExecutionStage {
  Activation(Int32, Int32, JointAudioAmpActivationPhase)
  Convolution(Int32, Int32, Int32, JointAudioAmpConvolutionPhase)
  Residual(Int32, Int32)
  ParallelAverage
} derive(Debug, Eq)

///|
pub(all) enum JointAudioAmpWorkspaceRole {
  OriginalUpsamplerOutput
  PrecomputedBlock0Activation0
  ActivationUpsampleWorkspace
  ActivationValueWorkspace
  ActivationOutputWorkspace
  NormalizedWeightWorkspace
  ConvolutionOutputWorkspace
  ResidualPingWorkspace
  ResidualPongWorkspace
  ParallelBlockOutput(Int32)
  FinalOutput
} derive(Debug, Eq)

///|
pub(all) enum JointAudioAmpStorageClass {
  ExternalInput
  Workspace
  ExternalOutput
} derive(Debug, Eq)

///|
pub(all) suberror JointAudioAmpWorkspaceLayoutError {
  InvalidGeometry
  InvalidStageCount
  InvalidStageOrder(Int32)
  InvalidRegionCount
  InvalidRegion(Int32)
  OverlappingLiveRegions(Int32, Int32)
  SizeOverflow
} derive(Debug, Eq)

///|
pub struct JointAudioAmpWorkspaceRegion {
  priv role : JointAudioAmpWorkspaceRole
  priv storage_class : JointAudioAmpStorageClass
  priv byte_offset : Int64
  priv byte_count : Int64
  priv live_start : Int32
  priv live_end : Int32
}

///|
pub struct JointAudioAmpWorkspaceLayoutDigest {
  priv hex : String
} derive(Debug, Eq)

///|
pub struct JointAudioAmpWorkspaceLayout {
  priv stages : ReadOnlyArray[JointAudioAmpExecutionStage]
  priv regions : ReadOnlyArray[JointAudioAmpWorkspaceRegion]
  priv workspace_bytes : Int64
  priv digest : JointAudioAmpWorkspaceLayoutDigest
}
