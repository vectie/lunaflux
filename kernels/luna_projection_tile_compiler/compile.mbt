///|
fn projection_tile_sha256(bytes : Bytes) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
pub fn select_projection_tile(
  request : ProjectionTileCompilerRequest,
) -> SelectedProjectionTileCompilation {
  { request, }
}

///|
pub fn compile_projection_tile(
  request : ProjectionTileCompilerRequest,
) -> ScheduledProjectionTileCompilation {
  schedule_projection_tile(
    optimize_projection_tile(
      elaborate_projection_tile(select_projection_tile(request)),
    ),
  )
}
