///|
/// Startup only. All ranks capture identical collective sites/counts. This
/// explicit schedule reduces the capacity of independent rows, not only live
/// rows; no consumer may observe padding. Initialize once before capture so
/// inactive operands are defined. Runtime counts still mask model/KV writes.
fn capture_collective_frame(
  queues : ArrayView[PreparedQueue],
  plan : @parallel.OrderedSubmissionPlan,
  bucket : Int,
  stream : @device.Stream,
  boundaries : ArrayView[ReductionBoundary],
  owner : @device.TensorCollectiveOwner,
  generation : UInt64,
  rows : Int,
  timeout_millis : UInt64,
) -> Unit raise @device.DeviceError {
  for boundary in boundaries {
    let bytes = boundary.row_count(rows).to_int64() *
      boundary.local_width.to_int64() *
      boundary.element_bytes()
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
  for window, queue in queues {
    let (first, end, first_site, end_site) = plan.collective_window(
      bucket, window,
    ) catch {
      _ => raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
    }
    queue.begin_effect_capture()
    errdefer (queue.discard_effect_graph() catch { _ => () })
    // The enclosing owner destroys ALL previously captured bucket graphs
    // before aborting the shared communicator and draining operand leases.
    let mut next = first
    for site in first_site..<end_site {
      let boundary = boundaries[site]
      while next < boundary.before_kernel {
        queue.enqueue(next - first)
        next += 1
      }
      queue.join_before(next - first)
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
    while next < end {
      queue.enqueue(next - first)
      next += 1
    }
    queue.end_effect_capture()
  }
  owner.seal_graph(0UL)
}

///|
fn ReductionBoundary::element_bytes(self : Self) -> Int64 {
  match self.kind {
    Bf16SumAllReduce | Bf16AllGather => 2L
    F32SumAllReduce | F32AllGather => 4L
  }
}
