///|
fn text_capture_boundaries(
  counts : ArrayView[Int],
) -> Array[(String, Int)] raise @device.DeviceError {
  let boundaries = [("context.bf16", 1)]
  for index, count in counts {
    guard count > 0 && count <= 2147483646 - boundaries.last().unwrap().1 else {
      raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
    }
    boundaries.push(
      ("refiner-\{index}.bf16", boundaries.last().unwrap().1 + count),
    )
  }
  boundaries.push(("conditioning.bf16", boundaries.last().unwrap().1 + 1))
  boundaries
}

///|
/// Offline observation of the prepared, unchanged conditioning queue. This
/// retires each actual stage before copying its output. Normal execute() has
/// no observation branches, extra waits, buffers or host readbacks.
pub fn TextConditioningProgram::capture_stages(
  self : Self,
  stream : @device.Stream,
  layers : ArrayView[
    (@minimax_h3_weights.RefinerWeightPacking, @device.Allocation),
  ],
) -> Array[(String, Bytes)] raise @device.DeviceError {
  guard self.phase == Prepared &&
    self.queue is Some(queue) &&
    self.output is Some(output) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  let counts = layers.map(pair => {
    self.refiner
    .bind_layer(pair.0, pair.1, state=output, state_offset=0L)
    .length()
  })
  let boundaries = text_capture_boundaries(counts)
  guard boundaries.last().unwrap().1 == queue.kernel_count() else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let snapshots : Array[(String, Bytes)] = []
  let previous = self.phase
  self.phase = Failed
  self.phase = execute_conditioning_once(previous, () => {
    observe_conditioning_stages(
      boundaries[:],
      index => queue.enqueue(index),
      () => stream.synchronize(),
      () => {
        queue.record_completion()
        queue.wait_completion()
      },
      label => {
        snapshots.push((label, output.copy_to_host(0L, output.byte_count())))
      },
    )
  })
  snapshots
}

///|
fn observe_conditioning_stages(
  boundaries : ArrayView[(String, Int)],
  enqueue : (Int) -> Unit raise @device.DeviceError,
  synchronize : () -> Unit raise @device.DeviceError,
  finish : () -> Unit raise @device.DeviceError,
  snapshot : (String) -> Unit raise @device.DeviceError,
) -> Unit raise @device.DeviceError {
  let mut start = 0
  for ordinal, (label, end) in boundaries {
    for index in start..<end {
      enqueue(index)
    }
    // An ordered executor's completion event is valid only after its complete
    // launch list. Intermediate diagnostic snapshots instead retire the exact
    // bound stream, retaining the executor's next-index/ownership state.
    if ordinal + 1 == boundaries.length() {
      finish()
    } else {
      synchronize()
    }
    snapshot(label)
    start = end
  }
}

///|
test "conditioning frontiers derive unequal layer extents, not a fixed stride" {
  assert_eq(text_capture_boundaries([4, 9]), [
    ("context.bf16", 1),
    ("refiner-0.bf16", 5),
    ("refiner-1.bf16", 14),
    ("conditioning.bf16", 15),
  ])
  for counts in [[0], [-1], [2147483647], [4, 2147483646]] {
    try text_capture_boundaries(counts) catch {
      @device.DeviceError::InvalidArgument(OrderedExecutorCreate) => ()
      _ => fail("wrong frontier error")
    } noraise {
      _ => fail("invalid frontier admitted")
    }
  }
}

///|
test "conditioning observer never records executor completion on a prefix" {
  let events : Array[String] = []
  observe_conditioning_stages(
    [("context", 1), ("layer", 3), ("final", 4)],
    index => events.push("enqueue-\{index}"),
    () => events.push("stream-retired"),
    () => events.push("complete-queue-retired"),
    label => events.push("snapshot-" + label),
  )
  assert_eq(events, [
    "enqueue-0", "stream-retired", "snapshot-context", "enqueue-1", "enqueue-2",
    "stream-retired", "snapshot-layer", "enqueue-3", "complete-queue-retired", "snapshot-final",
  ])
}
