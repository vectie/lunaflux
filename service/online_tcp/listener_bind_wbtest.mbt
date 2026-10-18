///|
async test "listener startup settles cancellation before acquiring socket ownership" {
  let mut entered = false
  let mut bound = false
  @async.with_task_group(group => {
    let task = group.spawn(() => {
      entered = true
      let listener = bind_luna_online_tcp_listener(
        @socket.Addr::parse("127.0.0.1:0"),
      )
      defer listener.close()
      bound = true
    })
    @async.pause()
    assert_true(entered)
    task.cancel()
  })
  assert_false(bound)
}

///|
async test "listener startup publishes a usable socket without cancellation" {
  let listener = bind_luna_online_tcp_listener(
    @socket.Addr::parse("127.0.0.1:0"),
  )
  defer listener.close()
  let client = @socket.Tcp::connect(listener.addr)
  defer client.close()
  let (accepted, _) = listener.accept()
  accepted.close()
}
