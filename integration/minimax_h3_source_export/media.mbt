///|
/// Extend a complete export with an actual reference-audio encoder module.
/// Each shape stays in an independent translation unit, even if symbol names
/// coincide. Ordinals must be unique within this export.
pub fn SourceExport::with_reference_audio(
  self : Self,
  ordinal~ : Int,
  samples~ : Int,
  max_scratch_bytes~ : Int64,
) -> Self raise {
  guard ordinal >= 0 else { raise InvalidModule }
  let source = @media.AudioSource::new(samples~, max_scratch_bytes~)
  self.append(
    unit(
      "reference_audio_\{ordinal}",
      source.kernels.source,
      source.kernels.symbols,
      [
        ("samples", samples.to_int64()),
        ("latent_time", source.plan.latent_time.to_int64()),
      ],
    ),
  )
}

///|
pub fn SourceExport::with_reference_video(
  self : Self,
  ordinal~ : Int,
  frames~ : Int,
  height~ : Int,
  width~ : Int,
  image~ : Bool,
  max_tiles~ : Int,
  max_scratch_bytes~ : Int64,
) -> Self raise {
  guard ordinal >= 0 else { raise InvalidModule }
  let source = @media.VideoSource::new(
    frames~,
    height~,
    width~,
    image~,
    max_tiles~,
    max_scratch_bytes~,
  )
  self.append(
    unit(
      "reference_video_\{ordinal}",
      source.kernels.source,
      source.kernels.symbols,
      [
        ("frames", frames.to_int64()),
        ("height", height.to_int64()),
        ("width", width.to_int64()),
        ("image", if image { 1L } else { 0L }),
      ],
    ),
  )
}

///|
fn SourceExport::append(
  self : Self,
  entry : SourceUnit,
) -> Self raise ExportError {
  guard !self.units.any(unit => unit.name == entry.name) else {
    raise DuplicateModule
  }
  let units = self.units[:].to_owned()
  units.push(entry)
  { ..self, units: ReadOnlyArray::from_array(units), }
}
