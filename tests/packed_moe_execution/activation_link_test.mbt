///|
async test "activation link retires producer and DMA before consumer publication" {
  device_fixture_lane.acquire()
  defer device_fixture_lane.release()
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let source = context.allocate(16L)
  defer (source.close() catch { _ => () })
  let destination = context.allocate(16L)
  defer (destination.close() catch { _ => () })
  let original : FixedArray[Byte] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
  ]
  context.copy_from_fixed_host(source, original, 0L, 0L, 16L)
  let listener = @socket.TcpServer::TcpServer(
    @socket.Addr::parse("127.0.0.1:0"),
  )
  defer listener.close()
  let client = @socket.Tcp::connect(listener.addr)
  let (peer, _) = listener.accept()
  let budget = @activation.required_host_bytes(maximum_rows=2, row_bytes=8)
  assert_eq(budget, 120L)
  let polls = Ref(0)
  try
    @activation.ActivationLink::sender(
      client,
      maximum_rows=2,
      row_bytes=8,
      maximum_host_bytes=budget - 1L,
      timeout_millis=1000UL,
      producer_ready=() => true,
    )
  catch {
    @activation.ActivationLinkError::HostBudgetExceeded => ()
    error => raise error
  } noraise {
    _ => fail("ignored host memory budget")
  }
  let sender = @activation.ActivationLink::sender(
    client,
    maximum_rows=2,
    row_bytes=8,
    maximum_host_bytes=budget,
    timeout_millis=1000UL,
    producer_ready=() => {
      polls.val += 1
      polls.val >= 3
    },
  )
  defer (sender.close() catch { _ => () })
  let receiver = @activation.ActivationLink::receiver(
    peer,
    maximum_rows=2,
    row_bytes=8,
    maximum_host_bytes=budget,
    timeout_millis=1000UL,
  )
  defer (receiver.close() catch { _ => () })
  sender.prepare(context, source)
  receiver.prepare(context, destination)
  sender.begin(epoch=1UL, live_rows=2, local_now_millis=0UL)
  receiver.begin(epoch=1UL, live_rows=2, local_now_millis=0UL)
  assert_false(sender.poll(local_now_millis=0UL))
  assert_false(sender.poll(local_now_millis=0UL))
  assert_eq(polls.val, 2)
  let waits = blocking_syncs()
  let mut sent = false
  let mut received = false
  for _ in 0..<128 {
    if !sent {
      sent = sender.poll(local_now_millis=0UL)
    }
    if !received {
      received = receiver.poll(local_now_millis=0UL)
    }
    if sent && received {
      break
    }
    @async.sleep(1)
  }
  assert_true(sent && received)
  assert_eq(blocking_syncs(), waits)
  assert_eq(polls.val, 3)
  let output = FixedArray::make(16, b'\x00')
  context.copy_to_fixed_host(destination, output, 0L, 0L, 16L)
  assert_eq(output, original)
  // A shorter next batch transfers only its live rows, not capacity padding.
  context.copy_from_fixed_host(
    source,
    [42, 43, 44, 45, 46, 47, 48, 49],
    0L,
    0L,
    8L,
  )
  sender.begin(epoch=2UL, live_rows=1, local_now_millis=1UL)
  receiver.begin(epoch=2UL, live_rows=1, local_now_millis=1UL)
  sent = false
  received = false
  for _ in 0..<128 {
    if !sent {
      sent = sender.poll(local_now_millis=1UL)
    }
    if !received {
      received = receiver.poll(local_now_millis=1UL)
    }
    if sent && received {
      break
    }
    @async.sleep(1)
  }
  assert_true(sent && received)
  context.copy_to_fixed_host(destination, output, 0L, 0L, 16L)
  assert_eq(output, [
    42, 43, 44, 45, 46, 47, 48, 49, 9, 10, 11, 12, 13, 14, 15, 16,
  ])
  // A different expected step must not upload stale activations.
  sender.begin(epoch=3UL, live_rows=1, local_now_millis=2UL)
  receiver.begin(epoch=4UL, live_rows=1, local_now_millis=2UL)
  sent = false
  let mut rejected = false
  for _ in 0..<128 {
    if !sent {
      sent = sender.poll(local_now_millis=2UL)
    }
    try receiver.poll(local_now_millis=2UL) catch {
      @activation.ActivationLinkError::MismatchedStep => {
        rejected = true
        break
      }
      error => raise error
    } noraise {
      _ => ()
    }
    @async.sleep(1)
  }
  assert_true(rejected)
  sender.close()
  receiver.close()
  sender.close()
  receiver.close()
  source.close()
  destination.close()
  context.close()
  assert_eq(live_resources(), baseline)
}
