///|
fn tile_plane(
  rows : Int,
  columns : Int,
  tile_rows : Int,
  payload : Bool,
) -> @physical_tile_ir.TiledPlane {
  @physical_tile_ir.TiledPlane::new(
    rows~,
    columns~,
    tile_rows~,
    layout=@physical_tile_ir.OperandLayout::new(
      columns=if payload { 128 } else { 16 },
      vector=16,
      rotations=if payload { 8 } else { 1 },
      row_group=1,
    ).unwrap(),
  ).unwrap()
}

///|
fn tile_address(
  plane : @physical_tile_ir.TiledPlane,
  row : String,
  column : String,
) -> String {
  let (_, columns, rows, width) = plane.geometry()
  let (_, vector, rotations, group) = plane.layout().geometry()
  "((((\{row})/\{rows})*\{columns/width}ULL+(\{column})/\{width})*\{rows*width}ULL+((\{row})%\{rows})*\{width}+(((\{column})%\{width})^((((\{row})%\{rows})/\{group}%\{rotations})*\{vector})))"
}

///|
/// Startup-only bijective permutation. One expert scratch preserves compact
/// byte size, scale identity, global scale and numerical publication contracts.
fn fp4_tile_bank_source(
  plan : @precision_ir.ExpertMlpPrecision,
  tile_columns : Int,
  symbol : String,
) -> String raise {
  let (_, hidden, intermediate, _, _) = plan.geometry()
  let parts = []
  for
    i, (rows, inner) in [
      (intermediate, hidden),
      (intermediate, hidden),
      (hidden, intermediate),
    ] {
    let base = plan.weight_offsets()[i]
    let scales = @precision_ir.packed_buffer_layout(
        plan.weights()[i].representation(),
      )
      .scales_offset()
      .unwrap()
    let payload = tile_address(
      tile_plane(rows, inner / 2, tile_columns, true),
      "row",
      "col",
    )
    let scale = tile_address(
      tile_plane(rows, inner / 16, tile_columns, false),
      "row",
      "col",
    )
    parts.push(
      (
        $|if(index>=\{base}ULL&&index<\{base+rows.to_int64()*(inner/2).to_int64()}ULL) {
        $|  unsigned long long local=index-\{base}ULL;int row=local/\{inner/2},col=local%\{inner/2};destination=\{base}ULL+\{payload};
        $|} else if(index>=\{base+scales}ULL&&index<\{base+scales+rows.to_int64()*(inner/16).to_int64()}ULL) {
        $|  unsigned long long local=index-\{base+scales}ULL;int row=local/\{inner/16},col=local%\{inner/16};destination=\{base+scales}ULL+\{scale};
        $|}
      ),
    )
  }
  (
    $|extern "C" __global__ void \{symbol}_prepare_bank(const unsigned char *source,unsigned char *target) {
    $| for(unsigned long long index=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;index<\{plan.expert_stride()}ULL;index+=(unsigned long long)gridDim.x*blockDim.x) {
    $|   unsigned long long destination=index;
    $|   \{parts.join("\n")}
    $|   target[destination]=source[index];
    $| }
    $|}
    $|extern "C" __global__ void \{symbol}_prepare_bank_copy(const unsigned char *source,unsigned char *target) {
    $| for(unsigned long long index=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;index<\{plan.expert_stride()}ULL;index+=(unsigned long long)gridDim.x*blockDim.x) target[index]=source[index];
    $|}
  )
}
