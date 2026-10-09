///|
async fn tls_sockets() -> (@socket.Tcp, @socket.Tcp) {
  let server = @socket.TcpServer::TcpServer(@socket.Addr::parse("127.0.0.1:0"))
  defer server.close()
  let left = @socket.Tcp::connect(server.addr)
  errdefer left.close()
  let (right, _) = server.accept()
  (left, right)
}

///|
async fn test_server(
  socket : @socket.Tcp,
  binding : Bytes,
) -> AuthenticatedRemoteChannel {
  accept(
    socket,
    RemoteCredential::from_bytes(Bytes::make(32, b'x')),
    binding,
    certificate_pem_file="runtime/remote_tls/test_keys/cert.pem",
    private_key_pem_file="runtime/remote_tls/test_keys/key.pem",
    max_frame_bytes=128,
    timeout_millis=2000,
  )
}

///|
async fn test_client(
  socket : @socket.Tcp,
  credential : Bytes,
  binding : Bytes,
  host : String,
) -> AuthenticatedRemoteChannel {
  connect(
    socket,
    RemoteCredential::from_bytes(credential),
    binding,
    server_name=host,
    ca_pem_file="runtime/remote_tls/test_keys/cert.pem",
    max_frame_bytes=128,
    timeout_millis=2000,
  )
}

///|
async test "TLS authenticated binding permits framed full duplex and explicit close" {
  let (left, right) = tls_sockets()
  @async.with_task_group() <| tasks => {
    tasks.spawn_bg(() => {
      let server = test_server(right, b"rank1/generation7/deploymentA")
      defer server.close()
      let received = FixedArray::make(128, b'x')
      assert_eq(server.receive(received), 5)
      assert_eq(Bytes::makei(5, i => received[i]), b"hello")
      server.send(received, 5)
    })
    let client = test_client(
      left,
      Bytes::make(32, b'x'),
      b"rank1/generation7/deploymentA",
      "localhost",
    )
    defer client.close()
    client.send(FixedArray::makei(5, i => b"hello"[i]), 5)
    let received = FixedArray::make(128, b'x')
    assert_eq(client.receive(received), 5)
    assert_eq(Bytes::makei(5, i => received[i]), b"hello")
    assert_eq(received[5], b'x')
  }
}

///|
async test "wrong credential rank generation or hostname fails before channel publication" {
  for case in 0..<3 {
    let (left, right) = tls_sockets()
    @async.with_task_group() <| tasks => {
      tasks.spawn_bg(() => {
        try test_server(right, b"rank1/generation7") catch {
          _ => ()
        } noraise {
          channel => {
            channel.close()
            fail("unauthorized client accepted")
          }
        }
      })
      try
        test_client(
          left,
          Bytes::make(32, if case == 0 { b'y' } else { b'x' }),
          if case == 1 {
            b"rank0/generation8"
          } else {
            b"rank1/generation7"
          },
          if case == 2 {
            "wrong.invalid"
          } else {
            "localhost"
          },
        )
      catch {
        _ => ()
      } noraise {
        channel => {
          channel.close()
          fail("authentication bypass")
        }
      }
    }
  }
}

///|
async test "TLS malformed length closes stream without publishing partial bytes" {
  let (left, right) = tls_sockets()
  @async.with_task_group() <| tasks => {
    tasks.spawn_bg(() => {
      let server = test_server(right, b"rank1")
      defer server.close()
      server.tls.write(b"\xff\xff\xff\xff")
      @async.sleep(20)
    })
    let client = test_client(left, Bytes::make(32, b'x'), b"rank1", "localhost")
    defer client.close()
    let bytes = FixedArray::make(128, b'z')
    try client.receive(bytes) catch {
      RemoteTlsError::Frame => ()
      error => fail("\{Repr(error)}")
    } noraise {
      _ => fail("oversize frame accepted")
    }
    assert_false(client.is_open())
    assert_eq(bytes, FixedArray::make(128, b'z'))
  }
}
