///|
/// Compose existing device lowerings according to a pure learned-head plan.
/// The text egress sees one collapsed stream, never an unweighted substitute.
pub fn learned_text_io_source(
  plan : @precision.LearnedTextIoPrecision,
  symbol_prefix~ : String,
) -> Bytes raise {
  let text = text_io_source(plan.text(), symbol_prefix~)
  let (_, hidden, _, rows, _) = plan.text().geometry()
  let reduction = @head.head_reduce_v1_bf16_f32_source(
    stream_width=plan.stream_count(),
    hidden_width=hidden,
    maximum_rows=rows,
    control_epsilon=plan.control_epsilon(),
    normalization_epsilon=plan.control_normalization_epsilon(),
    function_symbol=symbol_prefix + "_stream_reduce",
  )
  @utf8.encode(
    @utf8.decode(reduction.source_bytes()) +
    "\n" +
    @utf8.decode(text.source_bytes()),
  )
}
