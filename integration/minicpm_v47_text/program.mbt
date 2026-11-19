///|
pub using @decoder {type Phase}

///|
/// Text-to-logits/greedy plan for the flagship decoder. This retains the full
/// selected model's decoder; visual embedding construction is a separate owner.
pub struct Program {
  priv decoder : @decoder.Program
  priv text : @text_source.TextIoCudaSource
}

///|
pub fn Program::new(
  operators : @operators.DecoderOperators,
  decoder : @decoder.Program,
  maximum_output_rows~ : Int32 = 1,
  maximum_workspace_bytes~ : Int64,
) -> Self raise {
  let plan = operators.text_io(maximum_output_rows~, maximum_workspace_bytes~)
  guard plan.residual_bytes() == decoder.budget().activation_bytes else {
    fail("text/decoder row capacity mismatch")
  }
  {
    decoder,
    text: @text_source.text_io_source(
      plan,
      symbol_prefix="lunaflux_minicpm_v47_text",
    ),
  }
}

///|
pub fn Program::source_bytes(self : Self) -> Bytes {
  self.text.source_bytes()
}

///|
pub fn Program::text_source(self : Self) -> @text_source.TextIoCudaSource {
  self.text
}

///|
pub fn Program::decoder_program(self : Self) -> @decoder.Program {
  self.decoder
}

///|
pub fn Program::workspace_bytes(self : Self) -> Int64 {
  self.text.workspace_bytes()
}

///|
pub fn Program::launch_count(self : Self, phase : Phase) -> Int32 {
  self.decoder.launch_count(phase) + 1 + self.text.egress_order().length()
}
