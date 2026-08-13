///|
fn sift_down_int(values : FixedArray[Int], start : Int, length : Int) -> Unit {
  let mut root = start
  while root * 2 + 1 < length {
    let left = root * 2 + 1
    let right = left + 1
    let mut largest = left
    if right < length && values[right] > values[left] {
      largest = right
    }
    if values[root] >= values[largest] {
      break
    }
    let temporary = values[root]
    values[root] = values[largest]
    values[largest] = temporary
    root = largest
  }
}

///|
fn sort_int_prefix(values : FixedArray[Int], length : Int) -> Unit {
  if length <= 1 {
    return
  }
  let mut start = (length - 2) / 2
  for ;; {
    sift_down_int(values, start, length)
    if start == 0 {
      break
    }
    start = start - 1
  }
  let mut end = length - 1
  while end > 0 {
    let temporary = values[0]
    values[0] = values[end]
    values[end] = temporary
    sift_down_int(values, 0, end)
    end = end - 1
  }
}

///|
fn sift_down_u64(
  values : FixedArray[UInt64],
  start : Int,
  length : Int,
) -> Unit {
  let mut root = start
  while root * 2 + 1 < length {
    let left = root * 2 + 1
    let right = left + 1
    let mut largest = left
    if right < length && values[right] > values[left] {
      largest = right
    }
    if values[root] >= values[largest] {
      break
    }
    let temporary = values[root]
    values[root] = values[largest]
    values[largest] = temporary
    root = largest
  }
}

///|
fn sort_u64_prefix(values : FixedArray[UInt64], length : Int) -> Unit {
  if length <= 1 {
    return
  }
  let mut start = (length - 2) / 2
  for ;; {
    sift_down_u64(values, start, length)
    if start == 0 {
      break
    }
    start = start - 1
  }
  let mut end = length - 1
  while end > 0 {
    let temporary = values[0]
    values[0] = values[end]
    values[end] = temporary
    sift_down_u64(values, 0, end)
    end = end - 1
  }
}

///|
fn validate_unique_scalar_identities(
  requests : FixedArray[UInt64],
  slots : FixedArray[Int],
  count : Int,
) -> Unit raise WorkerWireError {
  sort_u64_prefix(requests, count)
  sort_int_prefix(slots, count)
  for index in 1..<count {
    if requests[index] == requests[index - 1] {
      raise InvalidFrame(RequestIdentity)
    }
    if slots[index] == slots[index - 1] {
      raise InvalidFrame(CompletionSlot)
    }
  }
}
