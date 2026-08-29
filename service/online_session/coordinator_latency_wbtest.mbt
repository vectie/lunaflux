///|
test "Luna max-one timing derives first inter-token and request intervals" {
  let service = coordinator_fixture().service
  let ticket : LunaOnlineRequestTicket = { lane: -1, epoch: 7UL }
  service.begin_request_timing_at(ticket, 100UL, true)
  let first = service.sample_request_timing_at(
    ticket,
    token=true,
    terminal=false,
    observed=125UL,
    available=true,
  )
  assert_eq(first.kind, 2)
  assert_eq(first.millis, 25)
  let next = service.sample_request_timing_at(
    ticket,
    token=true,
    terminal=false,
    observed=145UL,
    available=true,
  )
  assert_eq(next.kind, 3)
  assert_eq(next.millis, 20)
  let terminal = service.sample_request_timing_at(
    ticket,
    token=false,
    terminal=true,
    observed=180UL,
    available=true,
  )
  assert_eq(terminal.kind, 1)
  assert_eq(terminal.millis, 80)
  assert_eq(
    service.sample_request_timing_at(
      ticket,
      token=false,
      terminal=true,
      observed=181UL,
      available=true,
    ).kind,
    0,
  )
}

///|
test "Luna multi-lane timing is generation-authenticated and independent" {
  let service = coordinator_fixture(multi_lane_count=2).service
  let first_ticket : LunaOnlineRequestTicket = { lane: 0, epoch: 11UL }
  let second_ticket : LunaOnlineRequestTicket = { lane: 1, epoch: 12UL }
  service.begin_request_timing_at(first_ticket, 100UL, true)
  service.begin_request_timing_at(second_ticket, 200UL, true)
  let second_first = service.sample_request_timing_at(
    second_ticket,
    token=true,
    terminal=false,
    observed=215UL,
    available=true,
  )
  let first_first = service.sample_request_timing_at(
    first_ticket,
    token=true,
    terminal=false,
    observed=140UL,
    available=true,
  )
  assert_eq(second_first.millis, 15)
  assert_eq(first_first.millis, 40)
  let stale : LunaOnlineRequestTicket = { lane: 0, epoch: 10UL }
  service.retire_request_timing(stale)
  assert_eq(
    service.sample_request_timing_at(
      first_ticket,
      token=true,
      terminal=false,
      observed=150UL,
      available=true,
    ).millis,
    10,
  )
  assert_eq(
    service.sample_request_timing_at(
      second_ticket,
      token=false,
      terminal=true,
      observed=260UL,
      available=true,
    ).millis,
    60,
  )
}

///|
test "Luna timing rollback overflow and stale reuse fail closed" {
  let service = coordinator_fixture().service
  let first : LunaOnlineRequestTicket = { lane: 0, epoch: 21UL }
  service.begin_request_timing_at(first, 500UL, true)
  assert_eq(
    service.sample_request_timing_at(
      first,
      token=true,
      terminal=false,
      observed=499UL,
      available=true,
    ).kind,
    0,
  )
  assert_eq(
    service.sample_request_timing_at(
      first,
      token=true,
      terminal=false,
      observed=600UL,
      available=true,
    ).kind,
    0,
  )
  ignore(
    service.sample_request_timing_at(
      first,
      token=false,
      terminal=true,
      observed=700UL,
      available=true,
    ),
  )
  let second : LunaOnlineRequestTicket = { lane: 0, epoch: 22UL }
  service.begin_request_timing_at(second, 1UL, true)
  assert_eq(
    service.sample_request_timing_at(
      second,
      token=true,
      terminal=false,
      observed=0x80000001UL,
      available=true,
    ).kind,
    0,
  )
  service.retire_request_timing(first)
  assert_eq(service.timing_slots[1].generation, second.epoch)
  ignore(
    service.sample_request_timing_at(
      second,
      token=false,
      terminal=true,
      observed=2UL,
      available=false,
    ),
  )
  assert_eq(service.timing_slots[1].state, 0)
}
