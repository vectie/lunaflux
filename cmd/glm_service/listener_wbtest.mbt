///|
async test "rank listener rebinds a retired connection but rejects live ownership" {
  let listener = bind_rank_listener(@socket.Addr::parse("127.0.0.1:0"))
  let address = listener.addr
  try bind_rank_listener(address) catch {
    _ => ()
  } noraise {
    unexpected => {
      unexpected.close()
      listener.close()
      fail("two live TP listeners share an address")
    }
  }
  let client = @socket.Tcp::connect(address)
  let (peer, _) = listener.accept()
  peer.close()
  let buffer = FixedArray::make(1, b'\x00')
  assert_eq(client.read(buffer, offset=0, max_len=1), 0)
  client.close()
  listener.close()
  let rebound = bind_rank_listener(address)
  rebound.close()
}

///|
async test "rank listener pair releases the first listener if the second bind fails" {
  let vacant = bind_rank_listener(@socket.Addr::parse("127.0.0.1:0"))
  let first_address = vacant.addr
  vacant.close()
  let occupied = bind_rank_listener(@socket.Addr::parse("127.0.0.1:0"))
  defer occupied.close()
  try bind_rank_listeners("\{first_address}", "\{occupied.addr}") catch {
    _ => ()
  } noraise {
    (a, b) => {
      a.close()
      b.close()
      fail("occupied rendezvous bind passed")
    }
  }
  let recovered = bind_rank_listener(first_address)
  recovered.close()
}
