///|
async test "idle heartbeats renew both local leases while mailbox delivers exact bytes" {
  let listener = @socket.TcpServer::TcpServer(
    @socket.Addr::parse("127.0.0.1:0"),
  )
  defer listener.close()
  let credential = @remote_tls.RemoteCredential::from_bytes(
    Bytes::make(32, b'x'),
  )
  let worker_heartbeats = Ref(0UL)
  let parent_heartbeats = Ref(0UL)
  @async.with_task_group() <| tasks => {
    tasks.spawn_bg(() => {
      let (socket, _) = listener.accept()
      let channel = @remote_tls.accept(
        socket,
        credential,
        b"rank1/group9",
        certificate_pem_file="runtime/remote_tls/test_keys/cert.pem",
        private_key_pem_file="runtime/remote_tls/test_keys/key.pem",
        max_frame_bytes=129,
        timeout_millis=1000,
      )
      let mailbox = @remote_mailbox.RemoteMailbox::new(128)
      with_session(
        channel,
        mailbox,
        Worker,
        generation=9UL,
        lease_millis=300UL,
        heartbeat=sequence => worker_heartbeats.val = sequence,
        () => {
          let bytes = FixedArray::make(128, b'\x00')
          mailbox.begin_read()
          let length = for ;; {
            match mailbox.poll_read(bytes) {
              Some(length) => break length
              None => @async.sleep(1)
            }
          }
          assert_eq(length, 5)
          assert_eq(Bytes::makei(5, i => bytes[i]), b"hello")
          mailbox.begin_write(bytes, 0, length)
          while !mailbox.poll_write() {
            @async.sleep(1)
          }
          // Remain owned until the enclosing task cancels the worker.
          for ;; {
            @async.sleep(10)
          }
        },
      )
    })
    let socket = @socket.Tcp::connect(listener.addr)
    let channel = @remote_tls.connect(
      socket,
      credential,
      b"rank1/group9",
      server_name="localhost",
      ca_pem_file="runtime/remote_tls/test_keys/cert.pem",
      max_frame_bytes=129,
      timeout_millis=1000,
    )
    let mailbox = @remote_mailbox.RemoteMailbox::new(128)
    with_session(
      channel,
      mailbox,
      Coordinator,
      generation=9UL,
      lease_millis=300UL,
      heartbeat=sequence => parent_heartbeats.val = sequence,
      () => {
        // Exceeds the original lease without any model command traffic.
        @async.sleep(650)
        assert_true(parent_heartbeats.val >= 3UL)
        assert_true(worker_heartbeats.val >= parent_heartbeats.val)
        let bytes = FixedArray::makei(5, i => b"hello"[i])
        mailbox.begin_write(bytes, 0, 5)
        while !mailbox.poll_write() {
          @async.sleep(1)
        }
        mailbox.begin_read()
        while mailbox.poll_read(bytes) is None {
          @async.sleep(1)
        }
        assert_eq(Bytes::makei(5, i => bytes[i]), b"hello")
        tasks.return_immediately(())
      },
    )
  }
}

///|
test "heartbeat sequence decoder retains all sixty four bits" {
  let bytes = FixedArray::make(9, b'\xff')
  assert_eq(sequence(bytes), 0xffffffffffffffffUL)
  bytes[8] = b'\x00'
  assert_eq(sequence(bytes), 0x00ffffffffffffffUL)
}
