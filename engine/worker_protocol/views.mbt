///|
fn validate_range(
  start : Int,
  length : Int,
  capacity : Int,
  table : ProtocolTable,
) -> Unit raise WorkerProtocolError {
  if start < 0 {
    raise InvalidView(table~, rule=NegativeRangeStart)
  }
  if length <= 0 {
    raise InvalidView(table~, rule=EmptyRange)
  }
  if start > capacity || length > capacity - start {
    raise InvalidView(table~, rule=RangeOverflow)
  }
}

///|
/// Exact nonempty view into the flattened token table owned by a schedule.
#valtype
pub struct TokenRange(Int, Int) derive(Debug, Eq)

///|
pub fn TokenRange::new(
  start~ : Int,
  length~ : Int,
  limits : WorkerProtocolLimits,
) -> TokenRange raise WorkerProtocolError {
  validate_range(start, length, limits.max_plan_tokens(), Tokens)
  TokenRange(start, length)
}

///|
pub fn TokenRange::start(self : TokenRange) -> Int {
  self.0
}

///|
pub fn TokenRange::length(self : TokenRange) -> Int {
  self.1
}

///|
pub fn TokenRange::end(self : TokenRange) -> Int {
  self.0 + self.1
}

///|
/// Exact nonempty view into the flattened generational page table owned by a
/// schedule.
#valtype
pub struct PageTableView(Int, Int) derive(Debug, Eq)

///|
pub fn PageTableView::new(
  start~ : Int,
  length~ : Int,
  limits : WorkerProtocolLimits,
) -> PageTableView raise WorkerProtocolError {
  validate_range(start, length, limits.max_plan_pages(), Pages)
  PageTableView(start, length)
}

///|
pub fn PageTableView::start(self : PageTableView) -> Int {
  self.0
}

///|
pub fn PageTableView::length(self : PageTableView) -> Int {
  self.1
}

///|
pub fn PageTableView::end(self : PageTableView) -> Int {
  self.0 + self.1
}

///|
/// Exact nonempty view into the ordered capability table owned by a schedule.
#valtype
pub struct CapabilityView(Int, Int) derive(Debug, Eq)

///|
pub fn CapabilityView::new(
  start~ : Int,
  length~ : Int,
  limits : WorkerProtocolLimits,
) -> CapabilityView raise WorkerProtocolError {
  validate_range(start, length, limits.max_capabilities(), Capabilities)
  CapabilityView(start, length)
}

///|
pub fn CapabilityView::start(self : CapabilityView) -> Int {
  self.0
}

///|
pub fn CapabilityView::length(self : CapabilityView) -> Int {
  self.1
}

///|
pub fn CapabilityView::end(self : CapabilityView) -> Int {
  self.0 + self.1
}

///|
/// Bounded worker completion-table identity.
pub struct CompletionSlot {
  value : Int
} derive(Debug, Eq, Hash)

///|
pub fn CompletionSlot::new(
  value : Int,
  limits : WorkerProtocolLimits,
) -> CompletionSlot raise WorkerProtocolError {
  if value < 0 || value >= limits.max_completion_slots() {
    raise InvalidCompletionSlot
  }
  { value, }
}

///|
pub fn CompletionSlot::value(self : CompletionSlot) -> Int {
  self.value
}
