///|
/// Restart may bind after the previous connection closed into TIME_WAIT.
/// This never enables multiple live owners or load-balanced port sharing.
async fn bind_rank_listener(address : @socket.Addr) -> @socket.TcpServer {
  @socket.TcpServer::TcpServer(address, reuse_addr=true)
}

///|
async fn bind_rank_listeners(
  control : String,
  rendezvous : String,
) -> (@socket.TcpServer, @socket.TcpServer) {
  let first = bind_rank_listener(@socket.Addr::parse(control))
  errdefer first.close()
  let second = bind_rank_listener(@socket.Addr::parse(rendezvous))
  (first, second)
}
