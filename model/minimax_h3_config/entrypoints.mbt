///|
pub fn parse_pipeline_index(
  document : StringView,
  max_bytes~ : Int,
  max_depth~ : Int,
) -> MiniMaxH3PipelineIndex raise MiniMaxH3ConfigError {
  decode_pipeline(bounded_object(document, max_bytes, max_depth))
}

///|
pub fn parse_transformer(
  document : StringView,
  variant~ : @minimax_h3_spec.MiniMaxH3Variant,
  content_digest~ : @spec.ContentDigest,
  max_bytes~ : Int,
  max_depth~ : Int,
) -> @minimax_h3_spec.MiniMaxH3ModelMetadata raise MiniMaxH3ConfigError {
  decode_transformer(
    bounded_object(document, max_bytes, max_depth),
    variant,
    content_digest,
  )
}

///|
pub fn parse_video_vae(
  document : StringView,
  max_bytes~ : Int,
  max_depth~ : Int,
) -> MiniMaxH3VideoVaeConfig raise MiniMaxH3ConfigError {
  decode_video_vae(bounded_object(document, max_bytes, max_depth))
}

///|
pub fn parse_audio_vae(
  document : StringView,
  max_bytes~ : Int,
  max_depth~ : Int,
) -> MiniMaxH3AudioVaeConfig raise MiniMaxH3ConfigError {
  decode_audio_vae(bounded_object(document, max_bytes, max_depth))
}

///|
pub fn parse_scheduler(
  document : StringView,
  document_kind~ : MiniMaxH3ConfigDocument,
  max_bytes~ : Int,
  max_depth~ : Int,
) -> MiniMaxH3SchedulerConfig raise MiniMaxH3ConfigError {
  decode_scheduler(
    bounded_object(document, max_bytes, max_depth),
    document_kind,
  )
}
