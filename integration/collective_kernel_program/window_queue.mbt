///|
/// Existing IR validation has established that no collective cuts a fork.
/// Preserve both producer lanes and their join inside each effect window.
fn prepare_window_queue(
  context : @device.Context,
  stream : @device.Stream,
  launches : ArrayView[@device.OrderedKernelLaunch],
  forks : ArrayView[@parallel.ForkJoinSpan],
  first : Int,
  end : Int,
) -> PreparedQueue raise @device.DeviceError {
  let spans = forks.filter_map(span => {
    let (start, _, join) = span.range()
    if start >= first && join <= end {
      Some(
        span.shift(-first) catch {
          _ => abort("validated window fork coordinate")
        },
      )
    } else {
      None
    }
  })
  PreparedQueue::new(context, stream, launches[first:end], spans)
}
