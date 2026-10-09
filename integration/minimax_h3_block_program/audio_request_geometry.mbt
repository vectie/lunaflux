///|
/// H3 decoder strides multiply to 800. The reference preserves this rounded
/// latent duration; it only reinterprets [2,1,L] as [1,2,L], without cropping
/// to the independently rounded frame-duration sample count.
fn audio_join_geometry(
  latent_frames : Int,
  channels : Int,
  samples : Int,
  bytes : Int64,
) -> Bool {
  latent_frames > 0 &&
  channels == 2 &&
  samples.to_int64() == latent_frames.to_int64() * 800L &&
  bytes == samples.to_int64() * 8L
}

///|
/// Actual decoded sample count per waveform channel, not requested duration.
/// The canonical H3 output retains the rounded latent duration like upstream.
pub fn AudioDecodeRequest::waveform_samples(
  self : Self,
) -> Int raise @device.DeviceError {
  guard self.decoder is Some(decoder) && self.output is Some(_) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  decoder.tail.source.time
}
