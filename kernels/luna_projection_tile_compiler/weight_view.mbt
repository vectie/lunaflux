///|
/// Bind the logical dense-weight value to an ordered concatenation of row
/// operands. Empty bindings in an abstract request mean one unsplit weight.
/// Segmentation is semantic data, independent of device layout and scheduling.
/// Gated MLP has multiple weight roles and cannot use this single-role view.
pub fn ProjectionTileCompilerRequest::with_concatenated_weights(
  self : ProjectionTileCompilerRequest,
  row_counts : ArrayView[Int],
) -> ProjectionTileCompilerRequest raise ProjectionTileCompilerError {
  guard plan_projection_concatenated_rows(row_counts) is Some(segments) else {
    raise InvalidProjectionTileCompilation(InvalidProblem)
  }
  let (_, output_width, _) = self.dispatch.shape().dimensions()
  if self.dispatch.shape().family() == GatedMlp ||
    segments[segments.length() - 1].row_end != output_width {
    raise InvalidProjectionTileCompilation(InvalidProblem)
  }
  // An ingress epilogue observes Q/K/V separately: rebinding must preserve
  // the head partitions established by its validated constructor.
  if self.epilogue is ProjectionAttentionIngress(spec) {
    if row_counts.length() != 3 ||
      row_counts[0] != spec.query_heads * spec.head_dimension ||
      row_counts[1] != spec.kv_heads * spec.head_dimension ||
      row_counts[2] != spec.kv_heads * spec.head_dimension {
      raise InvalidProjectionTileCompilation(InvalidProblem)
    }
  }
  { ..self, concatenated_weights: segments, }
}

///|
/// Immutable operand row view of the logical dense weight. Preserved by
/// normalization; empty means an abstract, unsegmented weight binding.
pub fn ProjectionTileProgram::concatenated_weights(
  self : ProjectionTileProgram,
) -> ReadOnlyArray[ProjectionRowSegment] {
  self.concatenated_weights
}
