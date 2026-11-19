///|
pub extend JointVaeDecodeGap with @debug.Debug::{to_repr}

///|
pub extend JointVaeDecodeGap with Eq::{not_equal, equal}

///|
pub extend JointVaeDecodeReferenceError with @debug.Debug::{to_repr}

///|
pub extend JointVaeDecodeReferenceError with Eq::{not_equal, equal}

///|
pub extend JointVaeDecodeGapDigest with @debug.Debug::{to_repr}

///|
pub extend JointVaeDecodeGapDigest with Eq::{not_equal, equal}

///|
pub extend JointVaeLatentBuffer with @debug.Debug::{to_repr}

///|
pub extend JointVaeLatentBuffer with Eq::{not_equal, equal}

///|
pub extend JointVaeLatentDestandardizationDigest with @debug.Debug::{to_repr}

///|
pub extend JointVaeLatentDestandardizationDigest with Eq::{not_equal, equal}

///|
pub extend JointVaeAudioDecoderInputProjectionDigest with @debug.Debug::{
  to_repr,
}

///|
pub extend JointVaeAudioDecoderInputProjectionDigest with Eq::{not_equal, equal}

///|
pub extend JointVaeAudioDecoderConvPreDigest with @debug.Debug::{to_repr}

///|
pub extend JointVaeAudioDecoderConvPreDigest with Eq::{not_equal, equal}

///|
pub extend JointVaeAudioDecoderFirstUpsampleDigest with @debug.Debug::{to_repr}

///|
pub extend JointVaeAudioDecoderFirstUpsampleDigest with Eq::{not_equal, equal}

///|
pub extend JointVaeAudioDecoderUpsampleTransitionDigest with @debug.Debug::{
  to_repr,
}

///|
pub extend JointVaeAudioDecoderUpsampleTransitionDigest with Eq::{
  not_equal,
  equal,
}

///|
pub extend JointVaeAudioDecoderFirstAmpActivationDigest with @debug.Debug::{
  to_repr,
}

///|
pub extend JointVaeAudioDecoderFirstAmpActivationDigest with Eq::{
  not_equal,
  equal,
}

///|
pub extend JointVaeAudioDecoderAmpActivationDigest with @debug.Debug::{to_repr}

///|
pub extend JointVaeAudioDecoderAmpActivationDigest with Eq::{not_equal, equal}
