///|
priv enum HostState {
  Idle
  Ready
  Staged
  Submitted
  Finished
  Failed
}

///|
pub struct Host {
  priv profile : Profile
  priv packets : ReadOnlyArray[(Port, FixedArray[Byte])]
  priv canvas : FixedArray[Int32]
  priv causal : FixedArray[Int32]
  priv output : FixedArray[Int32]
  priv mut prompt : @prompt.PreparedTextPrompt?
  priv mut stops : ReadOnlyArray[Int32]
  priv mut state : HostState
  priv mut ceiling : Int32
  priv mut maximum_new : Int32
  priv mut cursor : Int32
  priv mut cached : Int32
  priv mut sampled : Int32
  priv mut emitted : Int32
  priv mut previous : Int32
  priv mut phase : Phase
  priv mut rows : Int32
  priv mut end_position : Int32
  priv mut needs_token : Bool
  priv mut reason : @inference.FinishReason?
}

///|
pub fn Host::new(profile : Profile) -> Self {
  let sizes : Array[(Port, Int32)] = [
    (Counts, 20),
    (TokenIds, profile.rows * 4),
    (CanvasPositions, profile.rows * 12),
    (CausalPositions, profile.rows * 4),
    (RowOffsets, 8),
    (SequenceLength, 4),
    (PageOffsets, 8),
    (PhysicalPages, profile.pages * 4),
    (RequestSlots, 4),
    (Reset, 4),
    (OutputCount, 4),
    (SelectedRows, 4),
    (SampledToken, 4),
  ]
  let packets = ReadOnlyArray::from_array(
    sizes.map(entry => (entry.0, FixedArray::make(entry.1, b'\x00'))),
  )
  let host = {
    profile,
    packets,
    canvas: FixedArray::make(profile.rows * 3, 0),
    causal: FixedArray::make(profile.rows, 0),
    output: FixedArray::make(profile.context, 0),
    prompt: None,
    stops: [],
    state: Idle,
    ceiling: profile.context,
    maximum_new: 0,
    cursor: 0,
    cached: 0,
    sampled: 0,
    emitted: 0,
    previous: -1,
    phase: Prefill,
    rows: 0,
    end_position: 0,
    needs_token: false,
    reason: None,
  }
  let pages = host.packet(PhysicalPages)
  for page in 0..<profile.pages {
    write_i32(pages, page, page)
  }
  host
}

///|
/// Borrowed packet, mutable only by the driver for sampled-token readback.
/// Other packet bytes remain read-only to callers until a new stage is built.
pub fn Host::packet(self : Self, port : Port) -> FixedArray[Byte] {
  for index in 0..<self.packets.length() {
    let (candidate, bytes) = self.packets[index]
    if candidate == port {
      return bytes
    }
  }
  abort("internal request packet")
}

///|
pub fn Host::begin(
  self : Self,
  prompt : @prompt.PreparedTextPrompt,
  maximum_new_tokens~ : Int32,
  stop_token_ids~ : ReadOnlyArray[Int32],
  context_ceiling? : Int32,
) -> Unit raise RequestError {
  guard self.state is (Idle | Finished | Failed) else { raise InvalidLifecycle }
  let ceiling = context_ceiling.unwrap_or(self.profile.context)
  guard maximum_new_tokens > 0 &&
    ceiling > 0 &&
    ceiling <= self.profile.context &&
    prompt.tokens().length() > 0 &&
    prompt.tokens().length() <= ceiling &&
    stop_token_ids.all(token => token >= 0 && token < self.profile.vocabulary) else {
    raise InvalidRequest
  }
  self.prompt = Some(prompt)
  self.stops = stop_token_ids
  self.ceiling = ceiling
  self.maximum_new = maximum_new_tokens
  self.cursor = 0
  self.cached = 0
  self.sampled = 0
  self.emitted = 0
  self.previous = -1
  self.rows = 0
  self.end_position = 0
  self.needs_token = false
  self.reason = None
  self.state = Ready
}

///|
pub fn Host::is_finished(self : Self) -> Bool {
  self.state is Finished
}

///|
pub fn Host::finish_reason(self : Self) -> @inference.FinishReason? {
  self.reason
}

///|
pub fn Host::sampled_tokens(self : Self) -> Int32 {
  self.sampled
}

///|
pub fn Host::emitted_tokens(self : Self) -> ArrayView[Int32] {
  self.output[:self.emitted]
}

///|
pub fn Host::cached_tokens(self : Self) -> Int32 {
  self.cached
}

///|
pub fn Host::staged_phase(self : Self) -> Phase raise RequestError {
  guard self.state is (Staged | Submitted) else { raise InvalidLifecycle }
  self.phase
}

///|
pub fn Host::expects_token(self : Self) -> Bool raise RequestError {
  guard self.state is (Staged | Submitted) else { raise InvalidLifecycle }
  self.needs_token
}

///|
pub fn Host::live_bytes(self : Self, port : Port) -> Int32 raise RequestError {
  guard self.state is (Staged | Submitted) else { raise InvalidLifecycle }
  match port {
    TokenIds | CausalPositions => self.rows * 4
    CanvasPositions => self.rows * 12
    _ => self.packet(port).length()
  }
}
