///|
pub(all) suberror PromptError {
  InvalidMedia
  InvalidPrompt
  InvalidTokenizer
  TokenBudgetExceeded
} derive(Eq)

///|
/// Already decoded/preprocessed media geometry; no paths, codecs or pixel IO.
/// Video timestamps describe temporal-patch blocks, not individual raw frames.
pub(all) enum Reference {
  Image(@text_encoder.VisualGrid)
  Audio
  Video(@text_encoder.VisualGrid, ReadOnlyArray[Double])
}

///|
pub(all) enum Presentation {
  Text
  Keyframes(ReadOnlyArray[(Int, @text_encoder.VisualGrid)])
  References(ReadOnlyArray[Reference])
}

///|
pub struct Prompt {
  token_ids : ReadOnlyArray[Int]
  token_tags : ReadOnlyArray[Int]
  grids : ReadOnlyArray[@text_encoder.VisualGrid]
  placement : @text_encoder.VisualPlacement?
}
