///|
fn LunaOnlineOpenAIConnectionSlot::begin_output_write(
  self : LunaOnlineOpenAIConnectionSlot,
) -> Bool {
  if !self.output.is_idle() ||
    self.output_writes.length() != 0 ||
    self.flights.length() != 0 {
    return false
  }
  let write = self.output.begin_write() catch { _ => return false }
  self.output_writes.push(write)
  self.response_copy_cursor = 0
  self.compat_copy_cursor = 0
  self.output_length = 0
  self.output_cursor = 0
  true
}

///|
fn LunaOnlineOpenAIConnectionSlot::publish_output(
  self : LunaOnlineOpenAIConnectionSlot,
) -> Bool {
  if self.output_writes.length() != 1 || self.output_length <= 0 {
    return false
  }
  let flight = self.output_writes[0].publish(self.output_length) catch {
    _ => return false
  }
  self.output_writes.clear()
  self.flights.push(flight)
  self.phase = LUNA_OPENAI_POOL_OUTPUT_WRITE
  true
}

///|
fn LunaOnlineOpenAIConnectionSlot::copy_response_view(
  self : LunaOnlineOpenAIConnectionSlot,
) -> Int {
  if self.response_views.length() != 1 || self.output_writes.length() != 1 {
    return -1
  }
  let length = self.response_views[0].length() catch { _ => return -1 }
  if length <= 0 ||
    self.response_copy_cursor < 0 ||
    self.response_copy_cursor >= length ||
    length - self.response_copy_cursor >
    self.output.byte_capacity() - self.output_length {
    return -1
  }
  let remaining = length - self.response_copy_cursor
  let copied = self.output_writes[0].copy_http_response_at(
    self.response_views[0],
    self.response_copy_cursor,
    self.output_length,
    remaining,
  ) catch {
    _ => return -1
  }
  if copied <= 0 || copied > remaining {
    return -1
  }
  self.response_copy_cursor += copied
  self.output_length += copied
  if self.response_copy_cursor != length {
    return 0
  }
  self.response_views[0].release() catch {
    _ => return -1
  }
  self.response_views.clear()
  self.response_copy_cursor = 0
  1
}

///|
fn LunaOnlineOpenAIConnectionSlot::copy_compat_view(
  self : LunaOnlineOpenAIConnectionSlot,
) -> Int {
  if self.outbound_views.length() != 1 ||
    self.semantic_events.length() != 1 ||
    self.output_writes.length() != 1 {
    return -1
  }
  let length = self.outbound_views[0].length() catch { _ => return -1 }
  if length <= 0 ||
    self.compat_copy_cursor < 0 ||
    self.compat_copy_cursor >= length ||
    length - self.compat_copy_cursor >
    self.output.byte_capacity() - self.output_length {
    return -1
  }
  let remaining = length - self.compat_copy_cursor
  let amount = Int::min(remaining, self.outbound_copy_quanta)
  if amount <= 0 {
    return -1
  }
  let copied = self.output_writes[0].copy_openai_response_at(
    self.outbound_views[0],
    self.compat_copy_cursor,
    self.output_length,
    amount,
  ) catch {
    _ => return -1
  }
  if copied <= 0 || copied > amount {
    return -1
  }
  self.compat_copy_cursor += copied
  self.output_length += copied
  if self.compat_copy_cursor != length {
    return 0
  }
  self.outbound_views[0].release() catch {
    _ => return -1
  }
  self.outbound_views.clear()
  self.semantic_events[0].delivered() catch {
    _ => return -1
  }
  self.semantic_events.clear()
  if self.event_terminal {
    self.terminal_observed = true
  }
  self.compat_copy_cursor = 0
  1
}

///|
fn luna_openai_pool_write_completes(
  cursor : Int,
  output_length : Int,
  written : Int,
) -> Bool {
  cursor >= 0 &&
  output_length > cursor &&
  written > 0 &&
  written <= output_length - cursor &&
  cursor + written == output_length
}

///|
fn LunaOnlineOpenAIConnectionPool::discard_semantic_event(
  self : LunaOnlineOpenAIConnectionPool,
  index : Int,
  event : @online_session.LunaOnlineFramedSemanticEvent,
  terminal : Bool,
) -> LunaOnlineOpenAIConnectionPoolProgress {
  let slot = self.slots[index]
  if slot.cancel_requested {
    event.discard_after_transport_cut() catch {
      _ => self.begin_drain()
    }
  } else {
    event.cancel_and_discard() catch {
      _ => {
        self.begin_drain()
        return LunaOnlineOpenAIConnectionPoolDrainingProgress
      }
    }
    slot.cancel_requested = true
  }
  if terminal {
    slot.terminal_observed = true
  }
  self.begin_slot_retirement(index)
  LunaOnlineOpenAIConnectionPoolAdvanced
}

///|
fn LunaOnlineOpenAIConnectionPool::capture_semantic_event(
  self : LunaOnlineOpenAIConnectionPool,
) -> LunaOnlineOpenAIConnectionPoolProgress {
  let event = self.stream().take_semantic_event() catch {
      _ => {
        self.begin_drain()
        return LunaOnlineOpenAIConnectionPoolDrainingProgress
      }
    }
  let route = event.request_route() catch {
    _ => {
      self.begin_drain()
      return LunaOnlineOpenAIConnectionPoolDrainingProgress
    }
  }
  let index = self.find_route_slot(route)
  if index < 0 {
    event.cancel_and_discard() catch {
      _ => self.begin_drain()
    }
    return LunaOnlineOpenAIConnectionPoolAdvanced
  }
  let slot = self.slots[index]
  let view = event.view() catch {
    _ => {
      self.begin_slot_retirement(index)
      return LunaOnlineOpenAIConnectionPoolAdvanced
    }
  }
  let kind = view.kind() catch {
    _ => {
      self.begin_slot_retirement(index)
      return LunaOnlineOpenAIConnectionPoolAdvanced
    }
  }
  let terminal = match kind {
    LunaCompletedEventKind | LunaFailedEventKind => true
    _ => false
  }
  if slot.phase == LUNA_OPENAI_POOL_RETIRING ||
    !slot.output.is_idle() ||
    slot.semantic_events.length() != 0 ||
    slot.outbound_works.length() != 0 ||
    slot.outbound_views.length() != 0 {
    return self.discard_semantic_event(index, event, terminal)
  }
  let mode : @openai_compat.LunaOpenAICompatMode = if slot.route_mode == 1 {
    LunaOpenAIResponses
  } else {
    LunaOpenAIChatCompletions
  }
  let work = slot.outbound_workspace.begin(
    view,
    mode,
    response_sequence=slot.request_sequence,
  ) catch {
    _ => return self.discard_semantic_event(index, event, terminal)
  }
  slot.event_terminal = terminal
  slot.semantic_events.push(event)
  slot.outbound_works.push(work)
  slot.phase = LUNA_OPENAI_POOL_COMPAT_PROGRESS
  LunaOnlineOpenAIConnectionPoolAdvanced
}

///|
fn LunaOnlineOpenAIConnectionPool::progress_slot_compat(
  self : LunaOnlineOpenAIConnectionPool,
  index : Int,
) -> LunaOnlineOpenAIConnectionPoolProgress {
  let slot = self.slots[index]
  if slot.outbound_works.length() != 1 || slot.semantic_events.length() != 1 {
    self.begin_slot_retirement(index)
    return LunaOnlineOpenAIConnectionPoolAdvanced
  }
  let state = slot.outbound_works[0].state() catch {
    _ => {
      self.begin_slot_retirement(index)
      return LunaOnlineOpenAIConnectionPoolAdvanced
    }
  }
  match state {
    LunaOpenAICompatReady => {
      let view = slot.outbound_works[0].take_view() catch {
        _ => {
          self.begin_slot_retirement(index)
          return LunaOnlineOpenAIConnectionPoolAdvanced
        }
      }
      slot.outbound_works.clear()
      slot.outbound_views.push(view)
      if !slot.begin_output_write() {
        self.begin_slot_retirement(index)
        return LunaOnlineOpenAIConnectionPoolAdvanced
      }
      if slot.sse_started {
        slot.phase = LUNA_OPENAI_POOL_RESPONSE_PROGRESS
        return LunaOnlineOpenAIConnectionPoolAdvanced
      }
      let response = slot.response_workspace.begin_event_stream() catch {
        _ => {
          self.begin_slot_retirement(index)
          return LunaOnlineOpenAIConnectionPoolAdvanced
        }
      }
      slot.response_kind = 1
      slot.response_works.push(response)
      slot.phase = LUNA_OPENAI_POOL_RESPONSE_PROGRESS
      LunaOnlineOpenAIConnectionPoolAdvanced
    }
    LunaOpenAICompatFailed => {
      self.begin_slot_retirement(index)
      LunaOnlineOpenAIConnectionPoolAdvanced
    }
    LunaOpenAICompatPending => {
      ignore(
        slot.outbound_works[0].progress() catch {
          _ => {
            self.begin_slot_retirement(index)
            return LunaOnlineOpenAIConnectionPoolAdvanced
          }
        },
      )
      LunaOnlineOpenAIConnectionPoolAdvanced
    }
  }
}

///|
fn LunaOnlineOpenAIConnectionPool::progress_slot_response(
  self : LunaOnlineOpenAIConnectionPool,
  index : Int,
) -> LunaOnlineOpenAIConnectionPoolProgress {
  let slot = self.slots[index]
  if slot.response_views.length() == 1 {
    match slot.copy_response_view() {
      -1 => self.begin_slot_retirement(index)
      0 => ()
      _ =>
        if slot.response_kind == 1 {
          slot.sse_started = true
        } else {
          let service_rejection = slot.response_kind == 4 ||
            slot.response_kind == 5
          if (service_rejection && slot.rejections.length() != 1) ||
            (!service_rejection && slot.rejections.length() != 0) ||
            slot.response_kind < 2 ||
            slot.response_kind > 5 {
            self.begin_slot_retirement(index)
            return LunaOnlineOpenAIConnectionPoolAdvanced
          }
          if service_rejection {
            slot.rejections[0].ack() catch {
              _ => {
                self.begin_drain()
                return LunaOnlineOpenAIConnectionPoolDrainingProgress
              }
            }
            slot.rejections.clear()
          }
          slot.terminal_observed = true
          if !slot.publish_output() {
            self.begin_slot_retirement(index)
          }
        }
    }
    return LunaOnlineOpenAIConnectionPoolAdvanced
  }
  if slot.response_works.length() == 1 {
    let state = slot.response_works[0].state() catch {
      _ => {
        self.begin_slot_retirement(index)
        return LunaOnlineOpenAIConnectionPoolAdvanced
      }
    }
    match state {
      LunaHttp1ResponseReady => {
        let view = slot.response_works[0].take_view() catch {
          _ => {
            self.begin_slot_retirement(index)
            return LunaOnlineOpenAIConnectionPoolAdvanced
          }
        }
        slot.response_works.clear()
        slot.response_views.push(view)
        if slot.output_writes.length() == 0 && !slot.begin_output_write() {
          self.begin_slot_retirement(index)
        }
        return LunaOnlineOpenAIConnectionPoolAdvanced
      }
      LunaHttp1ResponseBuilding => {
        ignore(
          slot.response_works[0].progress() catch {
            _ => {
              self.begin_slot_retirement(index)
              return LunaOnlineOpenAIConnectionPoolAdvanced
            }
          },
        )
        return LunaOnlineOpenAIConnectionPoolAdvanced
      }
    }
  }
  if slot.outbound_views.length() == 1 {
    match slot.copy_compat_view() {
      -1 => self.begin_slot_retirement(index)
      0 => ()
      _ => if !slot.publish_output() { self.begin_slot_retirement(index) }
    }
    return LunaOnlineOpenAIConnectionPoolAdvanced
  }
  self.begin_slot_retirement(index)
  LunaOnlineOpenAIConnectionPoolAdvanced
}

///|
async fn LunaOnlineOpenAIConnectionPool::write_slot_output(
  self : LunaOnlineOpenAIConnectionPool,
  index : Int,
) -> LunaOnlineOpenAIConnectionPoolProgress {
  let slot = self.slots[index]
  if slot.connections.length() != 1 || slot.flights.length() != 1 {
    self.begin_slot_retirement(index)
    return LunaOnlineOpenAIConnectionPoolAdvanced
  }
  if slot.output_cursor == slot.output_length {
    return self.release_completed_slot_output(index)
  }
  let length = Int::min(
    self.tcp_limits.write_chunk_bytes,
    slot.output_length - slot.output_cursor,
  )
  let written = @async.with_timeout_opt(
    self.pool_limits.io_poll_timeout_millis,
    () => {
      slot.flights[0].write_once_at(
        slot.connections[0],
        slot.output_cursor,
        length,
      )
    },
  ) catch {
    error if @async.is_being_cancelled() => raise error
    _ => {
      self.begin_slot_retirement(index)
      return LunaOnlineOpenAIConnectionPoolAdvanced
    }
  }
  match written {
    None => LunaOnlineOpenAIConnectionPoolBackpressured
    Some(count) => {
      if count <= 0 || count > length {
        self.begin_slot_retirement(index)
      } else {
        let completed = luna_openai_pool_write_completes(
          slot.output_cursor,
          slot.output_length,
          count,
        )
        slot.output_cursor += count
        if completed {
          return self.release_completed_slot_output(index)
        }
      }
      LunaOnlineOpenAIConnectionPoolAdvanced
    }
  }
}

///|
/// Releases the single output cell in the same bounded transition that writes
/// its last byte. A short or blocked write never reaches this transition and
/// therefore retains its flight until that connection is retired.
fn LunaOnlineOpenAIConnectionPool::release_completed_slot_output(
  self : LunaOnlineOpenAIConnectionPool,
  index : Int,
) -> LunaOnlineOpenAIConnectionPoolProgress {
  let slot = self.slots[index]
  if slot.flights.length() != 1 ||
    slot.output_length <= 0 ||
    slot.output_cursor != slot.output_length {
    self.begin_slot_retirement(index)
    return LunaOnlineOpenAIConnectionPoolAdvanced
  }
  slot.flights[0].release() catch {
    _ => {
      self.begin_drain()
      return LunaOnlineOpenAIConnectionPoolDrainingProgress
    }
  }
  slot.flights.clear()
  slot.output_cursor = 0
  slot.output_length = 0
  // Even a terminal write returns to the request-active phase first. Closing
  // the transport is a separate bounded transition, matching the reusable
  // server's write-confirmation/retirement sequencing.
  slot.phase = LUNA_OPENAI_POOL_REQUEST_ACTIVE
  LunaOnlineOpenAIConnectionPoolAdvanced
}
