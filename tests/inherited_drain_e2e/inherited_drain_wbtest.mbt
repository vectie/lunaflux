///|
test "fixed inherited descriptor is required and must be a Unix stream" {
  ignore(fixture_clear_fixed())
  try @drain.prepare_inherited_drain_v1() catch {
    InheritedDrainFailed(DescriptorUnavailable) => ()
    _ => abort("missing inherited drain descriptor returned wrong rule")
  } noraise {
    _ => abort("missing inherited drain descriptor was admitted")
  }

  let invalid = fixture_install(1)
  defer ignore(fixture_close(invalid))
  try @drain.prepare_inherited_drain_v1() catch {
    InheritedDrainFailed(InvalidDescriptor) => ()
    _ => abort("non-socket inherited descriptor returned wrong rule")
  } noraise {
    _ => abort("non-socket inherited descriptor was admitted")
  }
  ignore(fixture_clear_fixed())
}

///|
test "fragmented drain command confirms one fixed response and closes exactly" {
  let peer = fixture_install(0)
  defer ignore(fixture_close(peer))
  let preparation = @drain.prepare_inherited_drain_v1()
  assert_true(fixture_fixed_closed())
  let channel = preparation.activate()
  assert_eq(channel.progress(), InheritedDrainPending)
  let command = drain_command()
  assert_eq(fixture_send(peer, command, 0, 3), 3)
  assert_eq(channel.progress(), InheritedDrainPending)
  assert_eq(fixture_send(peer, command, 3, 5), 5)
  assert_eq(channel.progress(), InheritedDrainRequested)
  channel.begin_response(InheritedDrainAccepted)
  let mut attempts = 0
  while !channel.response_confirmed() && attempts < 1000 {
    ignore(channel.progress())
    attempts += 1
  }
  assert_true(channel.response_confirmed())
  assert_response(read_exact_response(peer), InheritedDrainAccepted)
  channel.close()
  assert_true(channel.is_closed())
}

///|
test "idempotent drain response uses a distinct fixed vocabulary" {
  let peer = fixture_install(0)
  defer ignore(fixture_close(peer))
  let channel = @drain.prepare_inherited_drain_v1().activate()
  let command = drain_command()
  assert_eq(fixture_send(peer, command, 0, command.length()), command.length())
  assert_eq(channel.progress(), InheritedDrainRequested)
  channel.begin_response(InheritedDrainAlreadyDraining)
  while !channel.response_confirmed() {
    ignore(channel.progress())
  }
  assert_response(read_exact_response(peer), InheritedDrainAlreadyDraining)
  channel.close()
}

///|
test "malformed trailing replay and partial channel loss never request drain" {
  for mode in 0..<3 {
    let peer = fixture_install(0)
    let channel = @drain.prepare_inherited_drain_v1().activate()
    let command = drain_command()
    if mode == 0 {
      command[0] = b'X'
      assert_eq(
        fixture_send(peer, command, 0, command.length()),
        command.length(),
      )
    } else if mode == 1 {
      assert_eq(
        fixture_send(peer, command, 0, command.length()),
        command.length(),
      )
      assert_eq(fixture_send(peer, command, 0, 1), 1)
    } else {
      assert_eq(fixture_send(peer, command, 0, 3), 3)
      assert_eq(fixture_close_write(peer), 0)
      assert_eq(channel.progress(), InheritedDrainPending)
    }
    let mut terminal = channel.progress()
    let mut attempts = 0
    while terminal == InheritedDrainPending && attempts < 1000 {
      terminal = channel.progress()
      attempts += 1
    }
    assert_eq(terminal, InheritedDrainChannelLost)
    assert_false(channel.request_received())
    assert_false(channel.is_live())
    ignore(fixture_close(peer))
  }
}

///|
test "replay after response is rejected without a second request" {
  let peer = fixture_install(0)
  defer ignore(fixture_close(peer))
  let channel = @drain.prepare_inherited_drain_v1().activate()
  let command = drain_command()
  assert_eq(fixture_send(peer, command, 0, 8), 8)
  assert_eq(channel.progress(), InheritedDrainRequested)
  channel.begin_response(InheritedDrainAccepted)
  while !channel.response_confirmed() {
    ignore(channel.progress())
  }
  assert_response(read_exact_response(peer), InheritedDrainAccepted)
  assert_eq(fixture_send(peer, command, 0, 8), 8)
  assert_eq(channel.progress(), InheritedDrainChannelLost)
  assert_true(channel.request_received())
}
