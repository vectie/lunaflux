///|
fn build_failure(bytes : Bytes) -> FixedArray[Int] {
  let failure = FixedArray::make(bytes.length(), 0)
  let mut matched = 0
  for index in 1..<bytes.length() {
    while matched > 0 && bytes[index] != bytes[matched] {
      matched = failure[matched - 1]
    }
    if bytes[index] == bytes[matched] {
      matched += 1
    }
    failure[index] = matched
  }
  failure
}

///|
fn build_patterns(stops : @inference.StopConditions) -> Array[StopPattern] {
  let patterns : Array[StopPattern] = []
  for value in stops.strings() {
    let bytes = @utf8.encode(value)
    patterns.push({ bytes, failure: build_failure(bytes) })
  }
  patterns
}

///|
fn StopPattern::advance(self : StopPattern, state : Int, byte : Byte) -> Int {
  let mut next = state
  while next > 0 && self.bytes[next] != byte {
    next = self.failure[next - 1]
  }
  if self.bytes[next] == byte {
    next += 1
  }
  next
}

///|
fn longest_pattern(patterns : Array[StopPattern]) -> Int {
  let mut longest = 0
  for pattern in patterns {
    if pattern.bytes.length() > longest {
      longest = pattern.bytes.length()
    }
  }
  longest
}
