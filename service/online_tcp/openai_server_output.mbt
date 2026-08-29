///|
fn LunaOnlineTcpOutputWrite::copy_http_response(
  self : LunaOnlineTcpOutputWrite,
  view : @http1.LunaHttp1ResponseView,
  source_offset : Int,
  length : Int,
) -> Int raise {
  let scratch = self.require_mutating()
  view.copy_chunk(
    scratch.mutable,
    destination_offset=0,
    source_offset~,
    length~,
  )
}

///|
fn LunaOnlineTcpOutputWrite::copy_http_response_at(
  self : LunaOnlineTcpOutputWrite,
  view : @http1.LunaHttp1ResponseView,
  source_offset : Int,
  destination_offset : Int,
  length : Int,
) -> Int raise {
  let scratch = self.require_mutating()
  view.copy_chunk(scratch.mutable, destination_offset~, source_offset~, length~)
}

///|
fn luna_openai_bounded_copy_length(
  remaining : Int,
  capacity : Int,
  work_quanta : Int,
) -> Int {
  if remaining <= 0 || capacity <= 0 || work_quanta <= 0 {
    return 0
  }
  Int::min(remaining, Int::min(capacity, work_quanta))
}

///|
fn luna_openai_terminal_confirmed(
  event_terminal : Bool,
  output_cursor : Int,
  output_length : Int,
) -> Bool {
  event_terminal && output_length > 0 && output_cursor == output_length
}

///|
fn LunaOnlineTcpOutputWrite::copy_openai_response(
  self : LunaOnlineTcpOutputWrite,
  view : @openai_compat.LunaOpenAICompatView,
  source_offset : Int,
  length : Int,
) -> Int raise {
  let scratch = self.require_mutating()
  view.copy_chunk_to(
    scratch.mutable,
    destination_offset=0,
    source_offset~,
    length~,
  )
}

///|
fn LunaOnlineTcpOutputWrite::copy_openai_response_at(
  self : LunaOnlineTcpOutputWrite,
  view : @openai_compat.LunaOpenAICompatView,
  source_offset : Int,
  destination_offset : Int,
  length : Int,
) -> Int raise {
  let scratch = self.require_mutating()
  view.copy_chunk_to(
    scratch.mutable,
    destination_offset~,
    source_offset~,
    length~,
  )
}

///|
fn LunaOnlineOpenAIServer::progress_response(
  self : LunaOnlineOpenAIServer,
) -> LunaOnlineTcpServerProgress {
  if !self.enforce_output_deadline() {
    return LunaOnlineTcpServerAdvanced
  }
  if self.response_works.length() != 1 {
    self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
    return LunaOnlineTcpServerAdvanced
  }
  let state = self.response_works[0].state() catch {
    _ => {
      self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
      return LunaOnlineTcpServerAdvanced
    }
  }
  match state {
    LunaHttp1ResponseReady => {
      let view = self.response_works[0].take_view() catch {
        _ => {
          self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
          return LunaOnlineTcpServerAdvanced
        }
      }
      self.response_works.clear()
      let length = view.length() catch {
        _ => {
          view.release() catch {
            _ => ()
          }
          self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
          return LunaOnlineTcpServerAdvanced
        }
      }
      self.response_views.push(view)
      self.output_source = 1
      self.output_cursor = 0
      self.output_length = length
      self.connection_phase = LUNA_OPENAI_RESPONSE_WRITE
      LunaOnlineTcpServerAdvanced
    }
    LunaHttp1ResponseBuilding => {
      ignore(
        self.response_works[0].progress() catch {
          _ => {
            self.begin_retirement(
              LunaLogInternal,
              failure=LunaOnlineTcpLifecycle,
            )
            return LunaOnlineTcpServerAdvanced
          }
        },
      )
      LunaOnlineTcpServerAdvanced
    }
  }
}

///|
fn LunaOnlineOpenAIServer::output_wait(
  self : LunaOnlineOpenAIServer,
) -> Int raise LunaOnlineTcpError {
  let remaining = if self.receipts.length() == 1 {
    self.receipts[0].remaining_millis() catch {
      _ => raise luna_online_tcp_error(LunaOnlineTcpTimeout)
    }
  } else {
    self.stream().transport_wait_remaining_millis() catch {
      _ => raise luna_online_tcp_error(LunaOnlineTcpCoordinator)
    }
  }
  luna_online_tcp_safe_wait(self.limits.write_timeout_millis, remaining)
}

///|
fn LunaOnlineOpenAIServer::enforce_output_deadline(
  self : LunaOnlineOpenAIServer,
) -> Bool {
  if self.receipts.length() == 1 {
    return self.enforce_receipt_deadline()
  }
  if self.streams.length() != 1 {
    self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
    return false
  }
  let remaining = self.stream().transport_wait_remaining_millis() catch {
      _ => {
        self.begin_retirement(LunaLogDeadline, failure=LunaOnlineTcpCoordinator)
        return false
      }
    }
  if remaining <= 0 {
    self.begin_retirement(LunaLogDeadline, failure=LunaOnlineTcpTimeout)
    return false
  }
  true
}

///|
fn LunaOnlineOpenAIServer::finish_output_view(
  self : LunaOnlineOpenAIServer,
) -> LunaOnlineTcpServerProgress {
  if self.output_source == 1 {
    self.response_views[0].release() catch {
      _ => {
        self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
        return LunaOnlineTcpServerAdvanced
      }
    }
    self.response_views.clear()
    self.output_source = 0
    self.output_cursor = 0
    self.output_length = 0
    if self.response_kind == 1 {
      self.connection_phase = LUNA_OPENAI_STREAM_PROGRESS
      self.sse_started = true
      if self.outbound_views.length() != 1 {
        self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
        return LunaOnlineTcpServerAdvanced
      }
      let length = self.outbound_views[0].length() catch {
        _ => {
          self.begin_retirement(
            LunaLogInternal,
            failure=LunaOnlineTcpCoordinator,
          )
          return LunaOnlineTcpServerAdvanced
        }
      }
      self.output_source = 2
      self.output_cursor = 0
      self.output_length = length
      self.connection_phase = LUNA_OPENAI_COMPAT_WRITE
      return LunaOnlineTcpServerAdvanced
    }
    if self.rejections.length() == 1 {
      self.rejections[0].ack() catch {
        _ => {
          self.begin_retirement(
            LunaLogInternal,
            failure=LunaOnlineTcpCoordinator,
          )
          return LunaOnlineTcpServerAdvanced
        }
      }
      self.rejections.clear()
    }
    let reason : @instance_log.LunaInstanceLogReason = match
      self.response_kind {
      2 => LunaLogAuthentication
      3 => LunaLogValidation
      4 => LunaLogCapacity
      _ => LunaLogInternal
    }
    self.begin_retirement(reason)
    return LunaOnlineTcpServerAdvanced
  }
  if self.output_source != 2 ||
    self.outbound_views.length() != 1 ||
    self.semantic_events.length() != 1 {
    self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
    return LunaOnlineTcpServerAdvanced
  }
  // The compat View is detached before the exact semantic delivery authority
  // is consumed; Stream progress performs the later lower ACK.
  self.outbound_views[0].release() catch {
    _ => {
      self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpCoordinator)
      return LunaOnlineTcpServerAdvanced
    }
  }
  self.outbound_views.clear()
  self.semantic_events[0].delivered() catch {
    _ => {
      self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpCoordinator)
      return LunaOnlineTcpServerAdvanced
    }
  }
  self.semantic_events.clear()
  if luna_openai_terminal_confirmed(
      self.event_terminal,
      self.output_cursor,
      self.output_length,
    ) {
    self.terminal_observed = true
  }
  self.output_source = 0
  self.output_cursor = 0
  self.output_length = 0
  self.connection_phase = LUNA_OPENAI_STREAM_PROGRESS
  LunaOnlineTcpServerAdvanced
}

///|
async fn LunaOnlineOpenAIServer::write_output_on_reactor(
  self : LunaOnlineOpenAIServer,
) -> LunaOnlineTcpServerProgress {
  if self.connections.length() != 1 ||
    self.flights.length() != 0 ||
    !self.output.is_idle() ||
    self.output_cursor < 0 ||
    self.output_cursor > self.output_length ||
    (self.output_source == 1 && self.response_views.length() != 1) ||
    (self.output_source == 2 && self.outbound_views.length() != 1) ||
    (self.output_source != 1 && self.output_source != 2) {
    self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
    return LunaOnlineTcpServerAdvanced
  }
  if self.output_cursor == self.output_length {
    return self.finish_output_view()
  }
  let copy_quanta = if self.output_source == 1 {
    self.response_copy_quanta
  } else {
    self.outbound_copy_quanta
  }
  let length = luna_openai_bounded_copy_length(
    self.output_length - self.output_cursor,
    self.output.byte_capacity(),
    copy_quanta,
  )
  let write = self.output.begin_write() catch {
    _ => {
      self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
      return LunaOnlineTcpServerAdvanced
    }
  }
  let copied = if self.output_source == 1 {
    write.copy_http_response(self.response_views[0], self.output_cursor, length) catch {
      _ => {
        write.abort() catch {
          _ => ()
        }
        self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
        return LunaOnlineTcpServerAdvanced
      }
    }
  } else {
    write.copy_openai_response(
      self.outbound_views[0],
      self.output_cursor,
      length,
    ) catch {
      _ => {
        write.abort() catch {
          _ => ()
        }
        self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
        return LunaOnlineTcpServerAdvanced
      }
    }
  }
  if copied != length {
    write.abort() catch {
      _ => ()
    }
    self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
    return LunaOnlineTcpServerAdvanced
  }
  let flight = write.publish(length) catch {
    _ => {
      write.abort() catch {
        _ => ()
      }
      self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
      return LunaOnlineTcpServerAdvanced
    }
  }
  self.flights.push(flight)
  let wait = self.output_wait() catch {
    _ => {
      self.begin_retirement(LunaLogDeadline, failure=LunaOnlineTcpTimeout)
      return LunaOnlineTcpServerAdvanced
    }
  }
  if wait <= 0 {
    self.begin_retirement(LunaLogDeadline, failure=LunaOnlineTcpTimeout)
    return LunaOnlineTcpServerAdvanced
  }
  let written = @async.with_timeout_opt(wait, () => {
    let count = self.flights[0].write_once(self.connections[0])
    if @async.is_being_cancelled() {
      @async.pause()
      raise luna_online_tcp_error(LunaOnlineTcpCancelled)
    }
    count
  }) catch {
    error if @async.is_being_cancelled() => {
      self.begin_retirement(LunaLogCancellation, failure=LunaOnlineTcpCancelled)
      raise error
    }
    _ => {
      self.begin_retirement(LunaLogNetworkWrite, failure=LunaOnlineTcpWrite)
      return LunaOnlineTcpServerAdvanced
    }
  }
  let count = match written {
    None => {
      self.begin_retirement(LunaLogDeadline, failure=LunaOnlineTcpTimeout)
      return LunaOnlineTcpServerAdvanced
    }
    Some(value) => value
  }
  if count <= 0 || count > length {
    self.begin_retirement(LunaLogNetworkWrite, failure=LunaOnlineTcpWrite)
    return LunaOnlineTcpServerAdvanced
  }
  // Positive write confirmation advances the exact compat/HTTP cursor before
  // the immutable Flight returns its backing to mutable state.
  self.output_cursor += count
  self.flights[0].release() catch {
    _ => {
      self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
      return LunaOnlineTcpServerAdvanced
    }
  }
  self.flights.clear()
  LunaOnlineTcpServerAdvanced
}
