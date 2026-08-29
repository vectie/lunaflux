///|
fn LunaOnlineFramedConnectionPool::retire_invalid_slot(
  self : LunaOnlineFramedConnectionPool,
  index : Int,
) -> LunaOnlineFramedConnectionPoolProgress {
  self.begin_slot_retirement(index)
  LunaOnlineFramedConnectionPoolAdvanced
}

///|
fn LunaOnlineFramedConnectionPool::offer_slot_request_tail(
  self : LunaOnlineFramedConnectionPool,
  index : Int,
) -> LunaOnlineFramedConnectionPoolProgress {
  let slot = self.slots[index]
  if slot.receipt_remaining() <= 0 ||
    slot.request_works.length() != 1 ||
    slot.input_tail_length <= 0 ||
    slot.input_tail_start < 0 ||
    slot.input_tail_start > slot.input_cursor - slot.input_tail_length {
    return self.retire_invalid_slot(index)
  }
  let consumed = slot.request_works[0].offer(
    slot.input,
    source_offset=slot.input_tail_start,
    length=slot.input_tail_length,
  ) catch {
    _ => return self.retire_invalid_slot(index)
  }
  if consumed < 0 || consumed > slot.input_tail_length {
    return self.retire_invalid_slot(index)
  }
  slot.input_tail_start += consumed
  slot.input_tail_length -= consumed
  if slot.input_tail_length == 0 {
    slot.input_tail_start = 0
  }
  let state = slot.request_works[0].state() catch {
    _ => return self.retire_invalid_slot(index)
  }
  match state {
    LunaFramedRequestFailed => self.retire_invalid_slot(index)
    LunaFramedRequestReady => {
      if slot.input_tail_length != 0 {
        return self.retire_invalid_slot(index)
      }
      self.finish_slot_request_validation(index)
    }
    LunaFramedRequestValidating => {
      slot.phase = LUNA_FRAMED_POOL_VALIDATING
      LunaOnlineFramedConnectionPoolAdvanced
    }
    LunaFramedRequestReceivingPrefix | LunaFramedRequestReceivingBody =>
      LunaOnlineFramedConnectionPoolAdvanced
  }
}

///|
async fn LunaOnlineFramedConnectionPool::read_slot_request(
  self : LunaOnlineFramedConnectionPool,
  index : Int,
) -> LunaOnlineFramedConnectionPoolProgress {
  let slot = self.slots[index]
  if slot.connections.length() != 1 ||
    slot.request_works.length() != 1 ||
    slot.input_tail_length != 0 ||
    slot.input_cursor < 0 ||
    slot.input_cursor >= slot.input.length() {
    return self.retire_invalid_slot(index)
  }
  let wait = luna_online_tcp_safe_wait(
    self.pool_limits.io_poll_timeout_millis,
    slot.receipt_remaining(),
  )
  if wait <= 0 {
    return self.retire_invalid_slot(index)
  }
  let amount = Int::min(
    self.tcp_limits.read_chunk_bytes,
    slot.input.length() - slot.input_cursor,
  )
  let received = @async.with_timeout_opt(wait, () => {
    slot.connections[0].read(
      slot.input,
      offset=slot.input_cursor,
      max_len=amount,
    )
  }) catch {
    error if @async.is_being_cancelled() => raise error
    _ => return self.retire_invalid_slot(index)
  }
  match received {
    None => LunaOnlineFramedConnectionPoolBackpressured
    Some(count) => {
      if count <= 0 || count > amount {
        return self.retire_invalid_slot(index)
      }
      slot.input_tail_start = slot.input_cursor
      slot.input_tail_length = count
      slot.input_cursor += count
      LunaOnlineFramedConnectionPoolAdvanced
    }
  }
}

///|
fn LunaOnlineFramedConnectionPool::finish_slot_request_validation(
  self : LunaOnlineFramedConnectionPool,
  index : Int,
) -> LunaOnlineFramedConnectionPoolProgress {
  let slot = self.slots[index]
  if slot.request_works.length() != 1 || slot.input_tail_length != 0 {
    return self.retire_invalid_slot(index)
  }
  let view = slot.request_works[0].take_view() catch {
    _ => return self.retire_invalid_slot(index)
  }
  slot.request_works.clear()
  slot.request_views.push(view)
  let length = view.length() catch {
    _ => return self.retire_invalid_slot(index)
  }
  if length != slot.input_cursor {
    return self.retire_invalid_slot(index)
  }
  view.release() catch {
    _ => return self.retire_invalid_slot(index)
  }
  slot.request_views.clear()
  slot.feed_cursor = 0
  slot.phase = LUNA_FRAMED_POOL_FEEDING
  LunaOnlineFramedConnectionPoolAdvanced
}

///|
fn LunaOnlineFramedConnectionPool::progress_slot_validation(
  self : LunaOnlineFramedConnectionPool,
  index : Int,
) -> LunaOnlineFramedConnectionPoolProgress {
  let slot = self.slots[index]
  if slot.receipt_remaining() <= 0 || slot.request_works.length() != 1 {
    return self.retire_invalid_slot(index)
  }
  ignore(
    slot.request_works[0].progress() catch {
      _ => return self.retire_invalid_slot(index)
    },
  )
  let state = slot.request_works[0].state() catch {
    _ => return self.retire_invalid_slot(index)
  }
  match state {
    LunaFramedRequestReady => self.finish_slot_request_validation(index)
    LunaFramedRequestFailed => self.retire_invalid_slot(index)
    LunaFramedRequestReceivingPrefix | LunaFramedRequestReceivingBody => {
      slot.phase = LUNA_FRAMED_POOL_RECEIVING
      LunaOnlineFramedConnectionPoolAdvanced
    }
    LunaFramedRequestValidating => LunaOnlineFramedConnectionPoolAdvanced
  }
}

///|
fn LunaOnlineFramedConnectionPool::progress_slot_feed(
  self : LunaOnlineFramedConnectionPool,
  index : Int,
) -> LunaOnlineFramedConnectionPoolProgress {
  let slot = self.slots[index]
  if self.ingress_slot >= 0 && self.ingress_slot != index {
    return LunaOnlineFramedConnectionPoolBackpressured
  }
  if (slot.routes.is_empty() && slot.receipt_remaining() <= 0) ||
    slot.feed_cursor < 0 ||
    slot.feed_cursor > slot.input_cursor {
    return self.retire_invalid_slot(index)
  }
  self.ingress_slot = index
  if slot.feed_cursor == slot.input_cursor {
    if slot.routes.length() == 1 {
      self.ingress_slot = -1
      slot.phase = LUNA_FRAMED_POOL_ACTIVE
      return LunaOnlineFramedConnectionPoolAdvanced
    }
    return LunaOnlineFramedConnectionPoolBackpressured
  }
  let ingress = if slot.receipts.length() == 1 {
    self
    .stream()
    .offer_luna_framed_with_receipt(
      slot.receipts[0],
      slot.input,
      source_offset=slot.feed_cursor,
      length=slot.input_cursor - slot.feed_cursor,
    ) catch {
      _ => return self.retire_invalid_slot(index)
    }
  } else {
    self
    .stream()
    .offer_luna_framed(
      slot.input,
      source_offset=slot.feed_cursor,
      length=slot.input_cursor - slot.feed_cursor,
    ) catch {
      _ => return self.retire_invalid_slot(index)
    }
  }
  let consumed = ingress.consumed_bytes()
  if consumed < 0 || consumed > slot.input_cursor - slot.feed_cursor {
    return self.retire_invalid_slot(index)
  }
  match ingress.kind() {
    LunaOnlineFramedIngressAccepted => {
      let route = ingress.request_route() catch {
        _ => return self.retire_invalid_slot(index)
      }
      if slot.routes.length() == 0 {
        slot.routes.push(route)
      } else if !slot.route().same_request(route) {
        return self.retire_invalid_slot(index)
      }
      if slot.receipts.length() == 1 {
        slot.receipts.clear()
      }
      slot.feed_cursor += consumed
      if slot.feed_cursor == slot.input_cursor {
        self.ingress_slot = -1
        slot.phase = LUNA_FRAMED_POOL_ACTIVE
      }
      LunaOnlineFramedConnectionPoolAdvanced
    }
    LunaOnlineFramedIngressBackpressured =>
      LunaOnlineFramedConnectionPoolBackpressured
    LunaOnlineFramedIngressDraining => {
      self.fail_and_drain()
      LunaOnlineFramedConnectionPoolFailedProgress
    }
  }
}
