///|
/// AOT prefill insertion; the full underlying text decoder/head are reused.
pub struct Program {
  priv text : @text.Program
  priv scatter : @scatter.EmbeddingScatterSource
}

///|
pub fn Program::new(text : @text.Program) -> Self raise {
  let (_, width, sequences, rows, outputs) = text
    .text_source()
    .plan()
    .geometry()
  let context = text.decoder_program().attention_program().context_tokens()
  guard sequences == 1 &&
    outputs == 1 &&
    width == 2048 &&
    @effects.EmbeddingScatterProgram::new(
      width~,
      maximum_rows=rows,
      maximum_positions=context,
      maximum_features=context,
      owners=128,
    )
    is Some(plan) else {
    fail("unsupported multimodal text profile")
  }
  {
    text,
    scatter: @scatter.embedding_scatter_source(
      plan,
      symbol="lunaflux_minicpm_v47_image_scatter",
    ),
  }
}

///|
pub fn Program::source_bytes(self : Self) -> Bytes {
  self.scatter.source_bytes()
}

///|
pub fn Program::scatter_source(self : Self) -> @scatter.EmbeddingScatterSource {
  self.scatter
}

///|
pub fn Program::text_program(self : Self) -> @text.Program {
  self.text
}

///|
pub fn Program::metadata_bytes(self : Self) -> Int64 {
  let (_, _, positions, _) = self.scatter.plan().geometry()
  positions.to_int64() * 4L + 4L
}

///|
pub fn Program::launch_count(self : Self, phase : @text.Phase) -> Int32 {
  self.text.launch_count(phase) + (if phase is Prefill { 1 } else { 0 })
}
