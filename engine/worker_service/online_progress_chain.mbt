///|
/// Reactor-only coalescing. The public single-transition API keeps its cuts
/// for diagnostic and cancellation callers. No loop, allocation, new flight,
/// publication acknowledgement or scheduler lookahead is introduced here.
fn OnlineWorkerLease::progress_online_chain(
  self : OnlineWorkerLease,
) -> WorkerServiceProgress raise WorkerServiceError {
  let first = self.progress()
  let next = match first {
    ServiceStarted(_) => self.progress()
    _ => first
  }
  match next {
    // CompletionReady transfers the exact retained flight internally. Commit
    // before yielding; cancellation already recorded in the scheduler remains
    // authoritative and backpressure keeps the accepted flight pinned.
    ServiceCompletionReady(_) => {
      let service = self.require_current()
      let committed = self.progress()
      if committed is ServiceCommitted(sequence) {
        self.record_stable_replacement_commit(service, sequence)
      }
      committed
    }
    _ => next
  }
}
