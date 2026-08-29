///|
priv struct LunaOnlineFramedTimingSlot {
  mut lane : Int
  mut generation : UInt64
  mut accepted_millis : UInt64
  mut last_token_millis : UInt64
  mut token_seen : Bool
  mut state : Int
}

///|
#valtype
priv struct LunaOnlineFramedTimingSample {
  kind : Int
  millis : Int
}

///|
fn new_luna_online_framed_timing_slots(
  lane_count : Int,
) -> FixedArray[LunaOnlineFramedTimingSlot] {
  FixedArray::makei(lane_count + 1, _ => {
    lane: -2,
    generation: 0UL,
    accepted_millis: 0UL,
    last_token_millis: 0UL,
    token_seen: false,
    state: 0,
  })
}

///|
fn LunaOnlineFramedService::timing_slot_index(
  self : LunaOnlineFramedService,
  ticket : LunaOnlineRequestTicket,
) -> Int {
  if ticket.epoch == 0UL {
    return -1
  }
  if ticket.lane == -1 {
    return 0
  }
  if ticket.lane < 0 || ticket.lane >= self.timing_slots.length() - 1 {
    return -1
  }
  ticket.lane + 1
}

///|
fn LunaOnlineFramedService::begin_request_timing_at(
  self : LunaOnlineFramedService,
  ticket : LunaOnlineRequestTicket,
  observed : UInt64,
  available : Bool,
) -> Unit {
  let index = self.timing_slot_index(ticket)
  if index < 0 {
    return
  }
  let slot = self.timing_slots[index]
  let clean = slot.state == 0
  slot.lane = ticket.lane
  slot.generation = ticket.epoch
  slot.accepted_millis = observed
  slot.last_token_millis = 0UL
  slot.token_seen = false
  slot.state = if clean && available { 1 } else { 2 }
}

///|
fn LunaOnlineFramedService::begin_request_timing(
  self : LunaOnlineFramedService,
  ticket : LunaOnlineRequestTicket,
) -> Unit {
  let observed = self.clock.now_millis() catch {
    _ => {
      self.begin_request_timing_at(ticket, 0UL, false)
      return
    }
  }
  self.begin_request_timing_at(ticket, observed, true)
}

///|
fn no_luna_online_framed_timing_sample() -> LunaOnlineFramedTimingSample {
  { kind: 0, millis: 0 }
}

///|
fn LunaOnlineFramedTimingSlot::clear(self : LunaOnlineFramedTimingSlot) -> Unit {
  self.lane = -2
  self.generation = 0UL
  self.accepted_millis = 0UL
  self.last_token_millis = 0UL
  self.token_seen = false
  self.state = 0
}

///|
fn LunaOnlineFramedService::sample_request_timing_at(
  self : LunaOnlineFramedService,
  ticket : LunaOnlineRequestTicket,
  token~ : Bool,
  terminal~ : Bool,
  observed~ : UInt64,
  available~ : Bool,
) -> LunaOnlineFramedTimingSample {
  let index = self.timing_slot_index(ticket)
  if index < 0 {
    return no_luna_online_framed_timing_sample()
  }
  let slot = self.timing_slots[index]
  if slot.state == 0 ||
    slot.lane != ticket.lane ||
    slot.generation != ticket.epoch {
    return no_luna_online_framed_timing_sample()
  }
  if slot.state != 1 || !available {
    if terminal {
      slot.clear()
    } else {
      slot.state = 2
    }
    return no_luna_online_framed_timing_sample()
  }
  let base = if token && slot.token_seen {
    slot.last_token_millis
  } else {
    slot.accepted_millis
  }
  if observed < base {
    if terminal {
      slot.clear()
    } else {
      slot.state = 2
    }
    return no_luna_online_framed_timing_sample()
  }
  let elapsed = observed - base
  if elapsed > 0x7fffffffUL {
    if terminal {
      slot.clear()
    } else {
      slot.state = 2
    }
    return no_luna_online_framed_timing_sample()
  }
  let kind = if terminal { 1 } else if !slot.token_seen { 2 } else { 3 }
  if terminal {
    slot.clear()
  } else {
    slot.last_token_millis = observed
    slot.token_seen = true
  }
  { kind, millis: elapsed.to_int() }
}

///|
fn LunaOnlineFramedService::sample_request_timing(
  self : LunaOnlineFramedService,
  ticket : LunaOnlineRequestTicket,
  token~ : Bool,
  terminal~ : Bool,
) -> LunaOnlineFramedTimingSample {
  let observed = self.clock.now_millis() catch {
    _ =>
      return self.sample_request_timing_at(
        ticket,
        token~,
        terminal~,
        observed=0UL,
        available=false,
      )
  }
  self.sample_request_timing_at(
    ticket,
    token~,
    terminal~,
    observed~,
    available=true,
  )
}

///|
fn LunaOnlineFramedService::retire_request_timing(
  self : LunaOnlineFramedService,
  ticket : LunaOnlineRequestTicket,
) -> Unit {
  let index = self.timing_slot_index(ticket)
  if index < 0 {
    return
  }
  let slot = self.timing_slots[index]
  if slot.lane == ticket.lane && slot.generation == ticket.epoch {
    slot.clear()
  }
}

///|
fn LunaOnlineFramedService::clear_request_timing(
  self : LunaOnlineFramedService,
) -> Unit {
  for slot in self.timing_slots {
    slot.clear()
  }
}
