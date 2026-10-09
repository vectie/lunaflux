///|
pub fn ValidatedTextConditioning::new(
  content_digest~ : @spec.ContentDigest,
  token_count~ : Int,
) -> ValidatedTextConditioning raise JointDiffusionIoPlanError {
  if token_count <= 0 || token_count > 65536 {
    raise InvalidTextConditioning
  }
  { content_digest, token_count, }
}

///|
pub fn ValidatedTextConditioning::content_digest(
  self : Self,
) -> @spec.ContentDigest {
  self.content_digest
}

///|
pub fn ValidatedTextConditioning::token_count(self : Self) -> Int {
  self.token_count
}

///|
pub fn ValidatedImageInput::new(
  content_digest~ : @spec.ContentDigest,
  height~ : Int,
  width~ : Int,
) -> ValidatedImageInput raise JointDiffusionIoPlanError {
  if height <= 0 || width <= 0 || height > 32768 || width > 32768 {
    raise InvalidMediaDescriptor
  }
  { content_digest, height, width, }
}

///|
pub fn ValidatedImageInput::content_digest(self : Self) -> @spec.ContentDigest {
  self.content_digest
}

///|
pub fn ValidatedImageInput::height(self : Self) -> Int {
  self.height
}

///|
pub fn ValidatedImageInput::width(self : Self) -> Int {
  self.width
}

///|
pub fn ValidatedVideoInput::new(
  content_digest~ : @spec.ContentDigest,
  height~ : Int,
  width~ : Int,
  frame_count~ : Int,
  frames_per_second~ : Int,
) -> ValidatedVideoInput raise JointDiffusionIoPlanError {
  if height <= 0 ||
    width <= 0 ||
    height > 32768 ||
    width > 32768 ||
    frame_count <= 0 ||
    frame_count > 1000000 ||
    frames_per_second <= 0 ||
    frames_per_second > 1000 {
    raise InvalidMediaDescriptor
  }
  { content_digest, height, width, frame_count, frames_per_second, }
}

///|
pub fn ValidatedVideoInput::content_digest(self : Self) -> @spec.ContentDigest {
  self.content_digest
}

///|
pub fn ValidatedVideoInput::height(self : Self) -> Int {
  self.height
}

///|
pub fn ValidatedVideoInput::width(self : Self) -> Int {
  self.width
}

///|
pub fn ValidatedVideoInput::frame_count(self : Self) -> Int {
  self.frame_count
}

///|
pub fn ValidatedVideoInput::frames_per_second(self : Self) -> Int {
  self.frames_per_second
}

///|
pub fn ValidatedAudioInput::new(
  content_digest~ : @spec.ContentDigest,
  sample_count~ : Int64,
  sample_rate~ : Int,
  channel_count~ : Int,
) -> ValidatedAudioInput raise JointDiffusionIoPlanError {
  if sample_count <= 0L ||
    sample_count > 2147483647L ||
    sample_rate <= 0 ||
    sample_rate > 768000 ||
    channel_count <= 0 ||
    channel_count > 64 {
    raise InvalidMediaDescriptor
  }
  { content_digest, sample_count, sample_rate, channel_count, }
}

///|
pub fn ValidatedAudioInput::content_digest(self : Self) -> @spec.ContentDigest {
  self.content_digest
}

///|
pub fn ValidatedAudioInput::sample_count(self : Self) -> Int64 {
  self.sample_count
}

///|
pub fn ValidatedAudioInput::sample_rate(self : Self) -> Int {
  self.sample_rate
}

///|
pub fn ValidatedAudioInput::channel_count(self : Self) -> Int {
  self.channel_count
}

///|
pub fn JointKeyframeConditioning::new(
  position~ : JointKeyframePosition,
  image~ : ValidatedImageInput,
) -> JointKeyframeConditioning {
  { position, image, }
}

///|
pub fn JointKeyframeConditioning::position(
  self : Self,
) -> JointKeyframePosition {
  self.position
}

///|
pub fn JointKeyframeConditioning::image(self : Self) -> ValidatedImageInput {
  self.image
}
