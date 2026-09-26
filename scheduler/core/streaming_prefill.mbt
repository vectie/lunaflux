///|
fn Scheduler::streaming_prefill_chunk(
  self : Scheduler,
  requested : Int,
  available_pages : Int,
  tail_free_tokens : Int,
  workspace_budget : Int64,
) -> Int {
  guard self.streaming is Some(owner) else { return requested }
  if owner.shapes.length() == 0 {
    return requested
  }
  let selected = @streaming_policy.select_prefill(
    owner.shapes[:],
    remaining_tokens=requested,
    token_budget=requested,
    workspace_budget~,
    available_pages~,
    decode_reserve=self.capacity.emergency_decode_page_reserve(),
    tokens_per_page=self.capacity.tokens_per_page(),
    tail_free_tokens~,
  )
  if selected < 0 {
    0
  } else {
    owner.capability.shape_at(selected).tokens()
  }
}

///|
fn Scheduler::streaming_workspace_budget(self : Scheduler) -> Int64 {
  match self.streaming {
    Some(owner) => owner.capability.workspace_budget()
    None => 0L
  }
}

///|
fn Scheduler::streaming_workspace_cost(self : Scheduler, tokens : Int) -> Int64 {
  guard self.streaming is Some(owner) else { return 0L }
  for i in 0..<owner.capability.shape_count() {
    let shape = owner.capability.shape_at(i)
    if shape.tokens() == tokens {
      return shape.workspace_bytes()
    }
  }
  0L
}
