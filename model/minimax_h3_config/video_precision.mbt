///|
/// The released decoder's autocast region is IEEE FP16, not BF16. Weight
/// storage, normalization, register/residual state and the final head stay F32.
/// The text encoder's separate BF16 contract is unaffected.
pub fn video_decoder_autocast() -> @precision.AutocastFloat {
  Float16
}
