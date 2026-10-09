///|
priv struct Use {
  start : Int64
  end : Int64
  step : Int
  mode : AccessMode
}

///|
fn overlaps(a : Use, buffer : Buffer) -> Bool {
  a.start < buffer.offset + buffer.bytes && buffer.offset < a.end
}

///|
// Reads are admitted only after their whole byte range has been initialized.
// Earlier admitted readers also witness initialization of their covered range.
fn initialized(frontier : Array[Use], buffer : Buffer) -> Bool {
  let end = buffer.offset + buffer.bytes
  for cursor = buffer.offset; cursor < end; {
    let mut next = cursor
    for previous in frontier {
      if previous.start <= cursor && previous.end > next {
        next = previous.end
      }
    }
    if next == cursor {
      return false
    }
    continue next
  }
  true
}

///|
// A write supersedes only the overlapped bytes. Retaining the other pieces
// preserves hazards for partially aliased views without a per-byte bitmap.
fn update_frontier(
  frontier : Array[Use],
  buffer : Buffer,
  mode : AccessMode,
  step : Int,
) -> Array[Use] {
  let next : Array[Use] = []
  for previous in frontier {
    if mode.writes() && overlaps(previous, buffer) {
      if previous.start < buffer.offset {
        next.push({ ..previous, end: buffer.offset, })
      }
      let end = buffer.offset + buffer.bytes
      if previous.end > end {
        next.push({ ..previous, start: end, })
      }
    } else {
      next.push(previous)
    }
  }
  next.push({
    start: buffer.offset,
    end: buffer.offset + buffer.bytes,
    step,
    mode,
  })
  next
}
