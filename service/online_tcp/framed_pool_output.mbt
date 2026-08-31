///|
fn LunaOnlineTcpOutputWrite::copy_framed_event_view_at(
  self : LunaOnlineTcpOutputWrite,
  view : @framed_wire.LunaFramedEventView,
  source_offset : Int,
  destination_offset : Int,
  length : Int,
) -> Int raise {
  let scratch = self.require_mutating()
  if source_offset < 0 ||
    destination_offset < 0 ||
    length <= 0 ||
    destination_offset > scratch.capacity - length {
    raise LunaOnlineTcpScratchRange
  }
  view.copy_chunk_to(
    scratch.mutable,
    destination_offset~,
    source_offset~,
    length~,
  )
}

///|
fn luna_online_framed_pool_event_copy_amount(
  write_chunk_bytes : Int,
  event_work_units : Int,
  remaining : Int,
) -> Int {
  Int::min(remaining, Int::min(write_chunk_bytes, event_work_units))
}

///|
fn luna_online_framed_pool_discard_after_transport_cut(
  phase : Int,
  cancel_requested : Bool,
) -> Bool {
  cancel_requested || phase == LUNA_FRAMED_POOL_RETIRING
}

///|
fn LunaOnlineFramedConnectionPool::capture_semantic_event(
  self : LunaOnlineFramedConnectionPool,
) -> LunaOnlineFramedConnectionPoolProgress {
  let event = self.stream().take_semantic_event() catch {
      _ => {
        self.fail_and_drain()
        return LunaOnlineFramedConnectionPoolFailedProgress
      }
    }
  let route = event.request_route() catch {
    _ => {
      event.cancel_and_discard() catch {
        _ => ()
      }
      self.fail_and_drain()
      return LunaOnlineFramedConnectionPoolFailedProgress
    }
  }
  let index = self.find_route_slot(route)
  if index < 0 {
    event.cancel_and_discard() catch {
      _ => ()
    }
    self.fail_and_drain()
    return LunaOnlineFramedConnectionPoolFailedProgress
  }
  let slot = self.slots[index]
  if slot.phase != LUNA_FRAMED_POOL_ACTIVE ||
    !slot.output.is_idle() ||
    !slot.semantic_events.is_empty() ||
    !slot.event_works.is_empty() ||
    !slot.event_views.is_empty() {
    let discard_after_cut = luna_online_framed_pool_discard_after_transport_cut(
      slot.phase,
      slot.cancel_requested,
    )
    try {
      if discard_after_cut {
        event.discard_after_transport_cut()
      } else {
        event.cancel_and_discard()
      }
    } catch {
      _ => {
        self.fail_and_drain()
        return LunaOnlineFramedConnectionPoolFailedProgress
      }
    }
    slot.cancel_requested = true
    self.begin_slot_retirement(index)
    return LunaOnlineFramedConnectionPoolAdvanced
  }
  let view = event.view() catch {
    _ => {
      event.cancel_and_discard() catch {
        _ => ()
      }
      self.begin_slot_retirement(index)
      return LunaOnlineFramedConnectionPoolAdvanced
    }
  }
  let kind = view.kind() catch {
    _ => {
      event.cancel_and_discard() catch {
        _ => ()
      }
      self.begin_slot_retirement(index)
      return LunaOnlineFramedConnectionPoolAdvanced
    }
  }
  let terminal = match kind {
    LunaCompletedEventKind | LunaFailedEventKind => true
    _ => false
  }
  let work = slot.event_workspace.begin(view) catch {
    _ => {
      event.cancel_and_discard() catch {
        _ => ()
      }
      self.begin_slot_retirement(index)
      return LunaOnlineFramedConnectionPoolAdvanced
    }
  }
  slot.event_terminal = terminal
  slot.semantic_events.push(event)
  slot.event_works.push(work)
  slot.phase = LUNA_FRAMED_POOL_ENCODING
  LunaOnlineFramedConnectionPoolAdvanced
}

///|
fn LunaOnlineFramedConnectionPool::progress_slot_encoding(
  self : LunaOnlineFramedConnectionPool,
  index : Int,
) -> LunaOnlineFramedConnectionPoolProgress {
  let slot = self.slots[index]
  if slot.event_works.length() != 1 || slot.semantic_events.length() != 1 {
    return self.retire_invalid_slot(index)
  }
  let state = slot.event_works[0].state() catch {
    _ => return self.retire_invalid_slot(index)
  }
  match state {
    LunaFramedEventFailed => self.retire_invalid_slot(index)
    LunaFramedEventPending => {
      ignore(
        slot.event_works[0].progress() catch {
          _ => return self.retire_invalid_slot(index)
        },
      )
      LunaOnlineFramedConnectionPoolAdvanced
    }
    LunaFramedEventReady => {
      let view = slot.event_works[0].take_view() catch {
        _ => return self.retire_invalid_slot(index)
      }
      slot.event_works.clear()
      let write = slot.output.begin_write() catch {
        _ => {
          view.release() catch {
            _ => ()
          }
          return self.retire_invalid_slot(index)
        }
      }
      slot.event_views.push(view)
      slot.output_writes.push(write)
      slot.event_copy_cursor = 0
      slot.output_length = 0
      slot.output_cursor = 0
      slot.phase = LUNA_FRAMED_POOL_COPYING
      LunaOnlineFramedConnectionPoolAdvanced
    }
  }
}

///|
fn LunaOnlineFramedConnectionPool::progress_slot_event_copy(
  self : LunaOnlineFramedConnectionPool,
  index : Int,
) -> LunaOnlineFramedConnectionPoolProgress {
  let slot = self.slots[index]
  if slot.event_views.length() != 1 ||
    slot.output_writes.length() != 1 ||
    slot.semantic_events.length() != 1 {
    return self.retire_invalid_slot(index)
  }
  let total = slot.event_views[0].length() catch {
    _ => return self.retire_invalid_slot(index)
  }
  if total <= 0 ||
    total > slot.output.byte_capacity() ||
    slot.event_copy_cursor < 0 ||
    slot.event_copy_cursor >= total {
    return self.retire_invalid_slot(index)
  }
  let amount = luna_online_framed_pool_event_copy_amount(
    self.tcp_limits.write_chunk_bytes,
    self.pool_limits.event_work_units,
    total - slot.event_copy_cursor,
  )
  let copied = slot.output_writes[0].copy_framed_event_view_at(
    slot.event_views[0],
    slot.event_copy_cursor,
    slot.event_copy_cursor,
    amount,
  ) catch {
    _ => return self.retire_invalid_slot(index)
  }
  if copied <= 0 || copied > amount {
    return self.retire_invalid_slot(index)
  }
  slot.event_copy_cursor += copied
  slot.output_length += copied
  if slot.event_copy_cursor != total {
    return LunaOnlineFramedConnectionPoolAdvanced
  }
  slot.event_views[0].release() catch {
    _ => return self.retire_invalid_slot(index)
  }
  slot.event_views.clear()
  let flight = slot.output_writes[0].publish(slot.output_length) catch {
    _ => return self.retire_invalid_slot(index)
  }
  slot.output_writes.clear()
  slot.flights.push(flight)
  // The immutable per-connection output cell now owns the exact event bytes;
  // release the global semantic credit before any potentially blocked socket.
  slot.semantic_events[0].delivered() catch {
    _ => return self.retire_invalid_slot(index)
  }
  slot.semantic_events.clear()
  self.delivered_events += 1UL
  if slot.event_terminal {
    slot.terminal_delivered = true
  }
  slot.phase = LUNA_FRAMED_POOL_WRITING
  LunaOnlineFramedConnectionPoolAdvanced
}

///|
async fn LunaOnlineFramedConnectionPool::write_slot_output(
  self : LunaOnlineFramedConnectionPool,
  index : Int,
) -> LunaOnlineFramedConnectionPoolProgress {
  let slot = self.slots[index]
  if slot.connections.length() != 1 ||
    slot.flights.length() != 1 ||
    slot.output_cursor < 0 ||
    slot.output_cursor >= slot.output_length {
    return self.retire_invalid_slot(index)
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
    _ => return self.retire_invalid_slot(index)
  }
  match written {
    None => LunaOnlineFramedConnectionPoolBackpressured
    Some(count) => {
      if count <= 0 || count > length {
        return self.retire_invalid_slot(index)
      }
      slot.output_cursor += count
      if slot.output_cursor != slot.output_length {
        return LunaOnlineFramedConnectionPoolAdvanced
      }
      slot.flights[0].release() catch {
        _ => {
          self.fail_and_drain()
          return LunaOnlineFramedConnectionPoolFailedProgress
        }
      }
      slot.flights.clear()
      slot.output_cursor = 0
      slot.output_length = 0
      slot.event_copy_cursor = 0
      if slot.event_terminal {
        self.begin_slot_retirement(index)
      } else {
        slot.phase = LUNA_FRAMED_POOL_ACTIVE
      }
      LunaOnlineFramedConnectionPoolAdvanced
    }
  }
}
