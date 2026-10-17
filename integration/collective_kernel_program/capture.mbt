///|
/// Startup only. All ranks capture identical collective sites/counts. This
/// explicit schedule reduces the capacity of independent rows, not only live
/// rows; no consumer may observe padding. Initialize once before capture so
/// inactive operands are defined. Runtime counts still mask model/KV writes.
fn capture_collective_frame(
  queue : PreparedQueue,
  stream : @device.Stream,
  boundaries : ArrayView[ReductionBoundary],
  owner : @device.TensorCollectiveOwner,
  generation : UInt64,
  rows : Int,
  kernel_count : Int,
  timeout_millis : UInt64,
) -> Unit raise @device.DeviceError {
  for boundary in boundaries {
    let bytes = boundary.row_count(rows).to_int64() *
      boundary.local_width.to_int64() *
      4L
    guard bytes <= 2147483647L else {
      raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
    }
    boundary.buffer.copy_from_host(
      Bytes::make(bytes.to_int(), 0),
      0L,
      0L,
      bytes,
    )
  }
  queue.begin_effect_capture()
  errdefer (queue.discard_effect_graph() catch { _ => () })
  // The enclosing owner destroys ALL previously captured bucket graphs
  // before aborting the shared communicator and draining operand leases.
  let mut next = 0
  for site, boundary in boundaries {
    while next < boundary.before_kernel {
      queue.enqueue(next)
      next += 1
    }
    queue.join_before(next)
    owner.submit(
      generation~,
      plan_sequence=1UL,
      global_sequence=(site + 1).to_uint64(),
      site_sequence=site + 1,
      operation_id=boundary.operation_id,
      kind=boundary.kind,
      live_query_tokens=boundary.row_count(rows),
      send=boundary.buffer,
      send_offset=0L,
      receive=boundary.receive,
      receive_offset=0L,
      stream~,
    )
    // Lazy peer connection may take seconds on the first nonblocking call.
    // Poll count is not elapsed time. Startup-only monotonic budget; never
    // query/synchronize a capturing device stream, or read clocks in replay.
    let clock = @clock.MonotonicClock::system()
    let stamp = Ref(0UL)
    guard clock.read_into(stamp) else {
      raise @device.DeviceError::DriverFailure(TensorCollectiveExecute)
    }
    let started = stamp.val
    while !owner.defer_submitted() {
      guard clock.read_into(stamp) && stamp.val - started < timeout_millis else {
        raise @device.DeviceError::Busy(TensorCollectiveExecute)
      }
    }
  }
  while next < kernel_count {
    queue.enqueue(next)
    next += 1
  }
  queue.end_effect_capture()
  owner.seal_graph(0UL)
}
