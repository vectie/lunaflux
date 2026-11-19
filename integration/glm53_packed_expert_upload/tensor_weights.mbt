///|
fn slice_weight_rows(
  weight : @precision_materialize.PackedCheckpointWeight,
  start : Int32,
  count : Int32,
) -> @precision_materialize.PackedCheckpointWeight raise {
  let representation = weight.representation()
  let (_, columns) = representation.shape()
  weight.slice(
    @precision_ir.PackedMatrixSlice::new(
      representation,
      row_start=start,
      rows=count,
      column_start=0,
      columns~,
    ),
  )
}

///|
fn slice_weight_columns(
  weight : @precision_materialize.PackedCheckpointWeight,
  start : Int32,
  count : Int32,
) -> @precision_materialize.PackedCheckpointWeight raise {
  let representation = weight.representation()
  let (rows, _) = representation.shape()
  weight.slice(
    @precision_ir.PackedMatrixSlice::new(
      representation,
      row_start=0,
      rows~,
      column_start=start,
      columns=count,
    ),
  )
}
