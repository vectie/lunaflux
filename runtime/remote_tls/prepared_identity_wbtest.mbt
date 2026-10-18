///|
async test "prepared server identity outlives its owner and frames span TLS records" {
  let identity = RemoteServerIdentity::load(
    certificate_pem_file="runtime/remote_tls/test_keys/cert.pem",
    private_key_pem_file="runtime/remote_tls/test_keys/key.pem",
  )
  defer identity.close()
  let (left, right) = tls_sockets()
  let credential = RemoteCredential::from_bytes(Bytes::make(32, b'x'))
  @async.with_task_group() <| tasks => {
    tasks.spawn_bg(() => {
      let server = accept_prepared(
        right,
        credential,
        b"prepared",
        identity,
        max_frame_bytes=131072,
        timeout_millis=4000,
      )
      defer server.close()
      // The active native session retains the context, not the file paths.
      identity.close()
      identity.close()
      let received = FixedArray::make(131072, b'\x00')
      for _ in 0..<3 {
        assert_eq(server.receive(received), 100003)
        for i in 0..<100003 {
          assert_eq(received[i], (i % 251).to_byte())
        }
        server.send(received, 100003)
      }
    })
    let client = connect(
      left,
      credential,
      b"prepared",
      server_name="localhost",
      ca_pem_file="runtime/remote_tls/test_keys/cert.pem",
      max_frame_bytes=131072,
      timeout_millis=4000,
    )
    defer client.close()
    let bytes = FixedArray::makei(100003, i => (i % 251).to_byte())
    for _ in 0..<3 {
      client.send(bytes, bytes.length())
      let received = FixedArray::make(131072, b'\xff')
      assert_eq(client.receive(received), bytes.length())
      for i in 0..<bytes.length() {
        assert_eq(received[i], bytes[i])
      }
      assert_eq(received[bytes.length()], b'\xff')
    }
  }
}

///|
async test "cancelled prepared handshake releases socket but identity remains reusable" {
  let identity = RemoteServerIdentity::load(
    certificate_pem_file="runtime/remote_tls/test_keys/cert.pem",
    private_key_pem_file="runtime/remote_tls/test_keys/key.pem",
  )
  defer identity.close()
  let credential = RemoteCredential::from_bytes(Bytes::make(32, b'x'))
  let (idle, pending) = tls_sockets()
  defer idle.close()
  try
    accept_prepared(
      pending,
      credential,
      b"reuse",
      identity,
      max_frame_bytes=128,
      timeout_millis=20,
    )
  catch {
    _ => ()
  } noraise {
    channel => {
      channel.close()
      fail("idle handshake completed")
    }
  }
  let (left, right) = tls_sockets()
  @async.with_task_group() <| tasks => {
    tasks.spawn_bg(() => {
      let server = accept_prepared(
        right,
        credential,
        b"reuse",
        identity,
        max_frame_bytes=128,
        timeout_millis=2000,
      )
      defer server.close()
      let received = FixedArray::make(128, b'\x00')
      assert_eq(server.receive(received), 1)
      server.send(received, 1)
    })
    let client = test_client(left, Bytes::make(32, b'x'), b"reuse", "localhost")
    defer client.close()
    client.send(FixedArray::make(1, b'x'), 1)
    let received = FixedArray::make(128, b'\x00')
    assert_eq(client.receive(received), 1)
    assert_eq(received[0], b'x')
  }
}
