///|
test "mailbox snapshots and waits for actual network send completion" {
  let mailbox = @remote_mailbox.RemoteMailbox::new(16)
  let source = FixedArray::make(4, b'a')
  let destination = FixedArray::make(16, b'x')
  mailbox.begin_write(source, 0, 4)
  source[0] = b'b'
  assert_false(mailbox.poll_write())
  assert_eq(mailbox.outgoing(destination), Some(4))
  assert_eq(destination[0], b'a')
  assert_false(mailbox.poll_write())
  mailbox.sent()
  assert_true(mailbox.poll_write())
  assert_eq(mailbox.outgoing(destination), None)
  mailbox.begin_read()
  assert_eq(mailbox.poll_read(destination), None)
  mailbox.deliver(source, 4)
  source[0] = b'c'
  assert_false(mailbox.can_deliver())
  assert_eq(mailbox.poll_read(destination), Some(4))
  assert_eq(destination[0], b'b')
  assert_eq(destination[4], b'x')
  assert_true(mailbox.can_deliver())
}

///|
test "failed mailbox never publishes pending bytes or synthetic completion" {
  let mailbox = @remote_mailbox.RemoteMailbox::new(16)
  let bytes = FixedArray::make(4, b'a')
  mailbox.begin_read()
  mailbox.deliver(bytes, 4)
  mailbox.begin_write(bytes, 0, 4)
  mailbox.fail()
  assert_false(mailbox.can_deliver())
  try mailbox.poll_read(bytes) catch {
    Closed => ()
    _ => fail("wrong error")
  } noraise {
    _ => fail("revived")
  }
  try mailbox.poll_write() catch {
    Closed => ()
    _ => fail("wrong error")
  } noraise {
    _ => fail("false send success")
  }
  try mailbox.deliver(bytes, 4) catch {
    Closed => ()
    _ => fail("wrong error")
  } noraise {
    _ => fail("revived")
  }
}
