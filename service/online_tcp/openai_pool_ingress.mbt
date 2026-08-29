///|
fn LunaOnlineOpenAIConnectionSlot::receipt_remaining(
  self : LunaOnlineOpenAIConnectionSlot,
) -> Int {
  if self.receipts.length() != 1 {
    return 0
  }
  self.receipts[0].remaining_millis() catch {
    _ => 0
  }
}

///|
fn LunaOnlineOpenAIConnectionSlot::begin_http_error(
  self : LunaOnlineOpenAIConnectionSlot,
  kind : @http1.LunaHttp1ErrorResponseKind,
) -> Bool {
  self.abort_local_authorities()
  let work = self.response_workspace.begin_error(kind) catch {
    _ => return false
  }
  self.response_works.push(work)
  self.response_kind = match kind {
    LunaHttp1AuthenticationError => 2
    LunaHttp1ValidationError => 3
    LunaHttp1CapacityError => 4
    LunaHttp1ServiceError => 5
  }
  self.phase = LUNA_OPENAI_POOL_RESPONSE_PROGRESS
  true
}

///|
fn LunaOnlineOpenAIConnectionPool::fail_slot_http(
  self : LunaOnlineOpenAIConnectionPool,
  index : Int,
  kind : @http1.LunaHttp1ErrorResponseKind,
) -> LunaOnlineOpenAIConnectionPoolProgress {
  if !self.slots[index].begin_http_error(kind) {
    self.begin_slot_retirement(index)
  }
  LunaOnlineOpenAIConnectionPoolAdvanced
}

///|
async fn LunaOnlineOpenAIConnectionPool::read_slot_http(
  self : LunaOnlineOpenAIConnectionPool,
  index : Int,
) -> LunaOnlineOpenAIConnectionPoolProgress {
  let slot = self.slots[index]
  if slot.connections.length() != 1 ||
    slot.tail_length != 0 ||
    (slot.route_mode == 3 && slot.embedded_control.requests.length() != 1) ||
    (slot.route_mode != 3 && slot.http_works.length() != 1) {
    self.begin_slot_retirement(index)
    return LunaOnlineOpenAIConnectionPoolAdvanced
  }
  let wait = if slot.route_mode == 3 {
    luna_online_http_control_poll_millis(
      self.pool_limits.io_poll_timeout_millis,
    )
  } else {
    luna_online_tcp_safe_wait(
      self.pool_limits.io_poll_timeout_millis,
      slot.receipt_remaining(),
    )
  }
  if wait <= 0 {
    self.begin_slot_retirement(index)
    return LunaOnlineOpenAIConnectionPoolAdvanced
  }
  let received = @async.with_timeout_opt(wait, () => {
    slot.connections[0].read(slot.input, offset=0, max_len=slot.input.length())
  }) catch {
    error if @async.is_being_cancelled() => raise error
    _ => {
      self.begin_slot_retirement(index)
      return LunaOnlineOpenAIConnectionPoolAdvanced
    }
  }
  match received {
    None => LunaOnlineOpenAIConnectionPoolBackpressured
    Some(count) => {
      if count <= 0 || count > slot.input.length() {
        self.begin_slot_retirement(index)
        return LunaOnlineOpenAIConnectionPoolAdvanced
      }
      slot.tail_start = 0
      slot.tail_length = count
      LunaOnlineOpenAIConnectionPoolAdvanced
    }
  }
}

///|
fn LunaOnlineOpenAIConnectionPool::offer_slot_http_tail(
  self : LunaOnlineOpenAIConnectionPool,
  index : Int,
) -> LunaOnlineOpenAIConnectionPoolProgress {
  let slot = self.slots[index]
  if slot.route_mode == 0 {
    if slot.tail_length <= 0 ||
      slot.tail_start < 0 ||
      slot.tail_start > slot.input.length() - slot.tail_length {
      self.begin_slot_retirement(index)
      return LunaOnlineOpenAIConnectionPoolAdvanced
    }
    if slot.input[slot.tail_start] == b'G' {
      slot.abort_local_authorities()
      if !slot.embedded_control.begin() {
        self.begin_slot_retirement(index)
        return LunaOnlineOpenAIConnectionPoolAdvanced
      }
      slot.route_mode = 3
      return self.offer_slot_embedded_control(index)
    }
    slot.route_mode = -1
  }
  if slot.route_mode == 3 {
    return self.offer_slot_embedded_control(index)
  }
  if slot.receipt_remaining() <= 0 ||
    slot.tail_length <= 0 ||
    slot.http_works.length() != 1 {
    self.begin_slot_retirement(index)
    return LunaOnlineOpenAIConnectionPoolAdvanced
  }
  let consumed = slot.http_works[0].offer(
    slot.input,
    source_offset=slot.tail_start,
    length=slot.tail_length,
  ) catch {
    _ => return self.fail_slot_http(index, LunaHttp1ValidationError)
  }
  if consumed < 0 || consumed > slot.tail_length {
    self.begin_slot_retirement(index)
    return LunaOnlineOpenAIConnectionPoolAdvanced
  }
  slot.tail_start += consumed
  slot.tail_length -= consumed
  if slot.tail_length == 0 {
    slot.tail_start = 0
  }
  slot.phase = LUNA_OPENAI_POOL_HTTP_PROGRESS
  LunaOnlineOpenAIConnectionPoolAdvanced
}

///|
fn LunaOnlineOpenAIConnectionPool::progress_slot_http(
  self : LunaOnlineOpenAIConnectionPool,
  index : Int,
) -> LunaOnlineOpenAIConnectionPoolProgress {
  let slot = self.slots[index]
  if slot.receipt_remaining() <= 0 || slot.http_works.length() != 1 {
    self.begin_slot_retirement(index)
    return LunaOnlineOpenAIConnectionPoolAdvanced
  }
  let state = slot.http_works[0].state() catch {
    _ => return self.fail_slot_http(index, LunaHttp1ValidationError)
  }
  match state {
    LunaHttp1Ready => {
      if slot.tail_length > 0 {
        return self.fail_slot_http(index, LunaHttp1ValidationError)
      }
      let view = slot.http_works[0].take_view() catch {
        _ => return self.fail_slot_http(index, LunaHttp1ValidationError)
      }
      slot.http_works.clear()
      let http_route = view.route() catch {
        _ => {
          view.release() catch {
            _ => ()
          }
          return self.fail_slot_http(index, LunaHttp1ValidationError)
        }
      }
      slot.route_mode = match http_route {
        LunaHttp1Responses => 1
        LunaHttp1ChatCompletions => 2
      }
      if self.next_request_sequence == 0xffffffffffffffffUL {
        view.release() catch {
          _ => ()
        }
        self.begin_drain()
        return LunaOnlineOpenAIConnectionPoolDrainingProgress
      }
      self.next_request_sequence += 1UL
      let request_id = @inference.RequestId::new(self.next_request_sequence) catch {
        _ => {
          view.release() catch {
            _ => ()
          }
          return self.fail_slot_http(index, LunaHttp1ValidationError)
        }
      }
      let inbound = slot.inbound_workspace.begin(view, request_id) catch {
        _ => {
          view.release() catch {
            _ => ()
          }
          return self.fail_slot_http(index, LunaHttp1ValidationError)
        }
      }
      slot.request_sequence = self.next_request_sequence
      slot.inbound_works.push(inbound)
      slot.phase = LUNA_OPENAI_POOL_INBOUND_PROGRESS
      LunaOnlineOpenAIConnectionPoolAdvanced
    }
    LunaHttp1Failed => {
      let failure = slot.http_works[0].failure() catch {
        _ => return self.fail_slot_http(index, LunaHttp1ValidationError)
      }
      let kind : @http1.LunaHttp1ErrorResponseKind = match failure {
        LunaHttp1Failed(rule=LunaHttp1Authorization, ..) =>
          LunaHttp1AuthenticationError
        _ => LunaHttp1ValidationError
      }
      self.fail_slot_http(index, kind)
    }
    state if luna_openai_http_requires_more_input(state) => {
      slot.phase = luna_openai_http_resume_phase(state)
      LunaOnlineOpenAIConnectionPoolAdvanced
    }
    _ => {
      ignore(
        slot.http_works[0].progress() catch {
          _ => return self.fail_slot_http(index, LunaHttp1ValidationError)
        },
      )
      LunaOnlineOpenAIConnectionPoolAdvanced
    }
  }
}

///|
fn LunaOnlineOpenAIConnectionPool::progress_slot_inbound(
  self : LunaOnlineOpenAIConnectionPool,
  index : Int,
) -> LunaOnlineOpenAIConnectionPoolProgress {
  let slot = self.slots[index]
  if slot.receipt_remaining() <= 0 || slot.inbound_works.length() != 1 {
    self.begin_slot_retirement(index)
    return LunaOnlineOpenAIConnectionPoolAdvanced
  }
  let state = slot.inbound_works[0].state() catch {
    _ => return self.fail_slot_http(index, LunaHttp1ValidationError)
  }
  match state {
    LunaOpenAIInboundReady => {
      let view = slot.inbound_works[0].take_view() catch {
        _ => return self.fail_slot_http(index, LunaHttp1ValidationError)
      }
      slot.inbound_works.clear()
      slot.inbound_views.push(view)
      slot.phase = LUNA_OPENAI_POOL_HANDOFF_FEED
      LunaOnlineOpenAIConnectionPoolAdvanced
    }
    LunaOpenAIInboundFailed =>
      self.fail_slot_http(index, LunaHttp1ValidationError)
    LunaOpenAIInboundPending => {
      ignore(
        slot.inbound_works[0].progress() catch {
          _ => return self.fail_slot_http(index, LunaHttp1ValidationError)
        },
      )
      LunaOnlineOpenAIConnectionPoolAdvanced
    }
  }
}

///|
fn LunaOnlineOpenAIConnectionPool::finish_slot_handoff(
  self : LunaOnlineOpenAIConnectionPool,
  index : Int,
) -> LunaOnlineOpenAIConnectionPoolProgress {
  let slot = self.slots[index]
  slot.inbound_views[0].release_after_transfer() catch {
    _ => {
      self.begin_slot_retirement(index)
      return LunaOnlineOpenAIConnectionPoolAdvanced
    }
  }
  slot.inbound_views.clear()
  slot.phase = LUNA_OPENAI_POOL_REQUEST_ACTIVE
  self.ingress_slot = -1
  LunaOnlineOpenAIConnectionPoolAdvanced
}

///|
fn LunaOnlineOpenAIConnectionPool::progress_slot_handoff(
  self : LunaOnlineOpenAIConnectionPool,
  index : Int,
) -> LunaOnlineOpenAIConnectionPoolProgress {
  let slot = self.slots[index]
  if self.ingress_slot >= 0 && self.ingress_slot != index {
    return LunaOnlineOpenAIConnectionPoolBackpressured
  }
  self.ingress_slot = index
  if slot.receipts.length() != 1 || slot.inbound_views.length() != 1 {
    self.begin_slot_retirement(index)
    return LunaOnlineOpenAIConnectionPoolAdvanced
  }
  let handoff = slot.inbound_views[0].handoff() catch {
    _ => {
      self.begin_slot_retirement(index)
      return LunaOnlineOpenAIConnectionPoolAdvanced
    }
  }
  let ingress = self
    .stream()
    .offer_luna_text_handoff_with_receipt(slot.receipts[0], handoff) catch {
      _ => {
        self.begin_slot_retirement(index)
        return LunaOnlineOpenAIConnectionPoolAdvanced
      }
    }
  let consumed = ingress.consumed_bytes()
  if !luna_openai_ingress_consumption_valid(ingress.kind(), consumed, 1) {
    self.begin_slot_retirement(index)
    return LunaOnlineOpenAIConnectionPoolAdvanced
  }
  match ingress.kind() {
    LunaOnlineFramedIngressAccepted => {
      let route = ingress.request_route() catch {
        _ => {
          self.begin_slot_retirement(index)
          return LunaOnlineOpenAIConnectionPoolAdvanced
        }
      }
      if slot.routes.length() == 0 {
        slot.routes.push(route)
      } else if !slot.route().same_request(route) {
        self.begin_slot_retirement(index)
        return LunaOnlineOpenAIConnectionPoolAdvanced
      }
      slot.receipts.clear()
      self.finish_slot_handoff(index)
    }
    LunaOnlineFramedIngressBackpressured =>
      LunaOnlineOpenAIConnectionPoolBackpressured
    LunaOnlineFramedIngressDraining => {
      self.begin_drain()
      LunaOnlineOpenAIConnectionPoolDrainingProgress
    }
  }
}
