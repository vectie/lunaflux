///|
/// A CUDA translation-unit fragment may end at a closing brace or a line
/// comment, without a final newline. Delimit fragments here rather than
/// relying on each renderer to terminate its text before the next directive.
fn projection_append_source_fragment(
  builder : StringBuilder,
  fragment : String,
) -> Unit {
  if !fragment.is_empty() {
    builder.write_char('\n')
    builder.write_string(fragment)
    if !fragment.has_suffix("\n") {
      builder.write_char('\n')
    }
  }
}

///|
fn projection_join_source_fragments(
  primary : Bytes,
  variants : String,
) -> Bytes raise ProjectionAotError {
  if variants.is_empty() {
    return primary
  }
  let builder = StringBuilder()
  builder.write_string(
    @utf8.decode(primary) catch {
      _ => raise Invalid(CompilerPolicy)
    },
  )
  projection_append_source_fragment(builder, variants)
  @utf8.encode(builder.to_string())
}
