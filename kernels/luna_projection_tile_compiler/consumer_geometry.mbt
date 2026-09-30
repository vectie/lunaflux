///|
/// A finite consumer-product choice changes who owns independent results,
/// never how the ordered reduction is evaluated. Device subgroup width stays
/// outside this pure plan; the terminal backend maps groups to launch threads.
fn projection_primary_consumer_groups(
  request : ProjectionTileCompilerRequest,
) -> Int {
  for choice in request.fold_choices {
    if (choice.role == SiblingFold || choice.role == MatrixFold) &&
      choice.consumer_groups is Some(groups) {
      return groups
    }
  }
  request.dispatch.prefill().output_tiles_per_workgroup()
}
