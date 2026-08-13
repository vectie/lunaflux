///|
pub fn WorkerService::admit(
  self : WorkerService,
  request : @core.TokenizedRequest,
  now_millis : UInt64,
) -> @core.RequestHandle raise WorkerServiceError {
  self.require_raw_mode()
  self.require_lifecycle(ServiceReady)
  self.scheduler.admit(request, now_millis) catch {
    error => raise scheduler_error(error)
  }
}

///|
pub fn WorkerService::cancel(
  self : WorkerService,
  handle : @core.RequestHandle,
) -> Unit raise WorkerServiceError {
  self.require_raw_mode()
  self.require_lifecycle(ServiceReady)
  self.scheduler.cancel(handle) catch {
    error => raise scheduler_error(error)
  }
}

///|
pub fn WorkerService::expire_deadlines(
  self : WorkerService,
  now_millis : UInt64,
) -> Int raise WorkerServiceError {
  self.require_raw_mode()
  self.require_lifecycle(ServiceReady)
  self.scheduler.expire_deadlines(now_millis) catch {
    error => raise scheduler_error(error)
  }
}

///|
pub fn WorkerService::status(
  self : WorkerService,
  request_id : @inference.RequestId,
) -> @core.RequestStatus? raise WorkerServiceError {
  self.require_raw_mode()
  self.scheduler.status(request_id)
}

///|
pub fn WorkerService::take_generated_token(
  self : WorkerService,
) -> @core.GeneratedTokenNotice raise WorkerServiceError {
  self.require_raw_mode()
  self.scheduler.take_generated_token() catch {
    error => raise scheduler_error(error)
  }
}

///|
pub fn WorkerService::take_terminal_notice(
  self : WorkerService,
) -> @core.TerminalNotice raise WorkerServiceError {
  self.require_raw_mode()
  self.scheduler.take_terminal_notice() catch {
    error => raise scheduler_error(error)
  }
}

///|
/// Returns which scheduler publication ring owns the globally oldest notice.
pub fn WorkerService::next_publication_kind(
  self : WorkerService,
) -> @core.SchedulerPublicationKind raise WorkerServiceError {
  self.require_raw_mode()
  self.scheduler.next_publication_kind() catch {
    error => raise scheduler_error(error)
  }
}

///|
/// Consumes the globally oldest scheduler publication without exposing a
/// caller-selected physical ring.
pub fn WorkerService::take_next_publication(
  self : WorkerService,
) -> @core.SchedulerPublication raise WorkerServiceError {
  self.require_raw_mode()
  self.scheduler.take_next_publication() catch {
    error => raise scheduler_error(error)
  }
}

///|
pub fn WorkerService::live_request_count(
  self : WorkerService,
) -> Int raise WorkerServiceError {
  self.require_raw_mode()
  self.scheduler.live_request_count()
}

///|
pub fn WorkerService::generated_token_count(
  self : WorkerService,
) -> Int raise WorkerServiceError {
  self.require_raw_mode()
  self.scheduler.generated_token_count()
}

///|
pub fn WorkerService::terminal_notice_count(
  self : WorkerService,
) -> Int raise WorkerServiceError {
  self.require_raw_mode()
  self.scheduler.terminal_notice_count()
}

///|
pub fn WorkerService::free_page_count(
  self : WorkerService,
) -> Int raise WorkerServiceError {
  self.require_raw_mode()
  self.scheduler.free_page_count()
}

///|
pub fn WorkerService::allocated_block_table_count(
  self : WorkerService,
) -> Int raise WorkerServiceError {
  self.require_raw_mode()
  self.scheduler.allocated_block_table_count()
}
