///|
/// Owns the single native listener bind policy for LunaFlux servers.
/// `SO_REUSEADDR` is required for deterministic same-address restart after a
/// server actively closes an authenticated or rejected connection. This does
/// not enable `SO_REUSEPORT`; a concurrently live exact-address listener still
/// retains exclusive bind ownership.
async fn bind_luna_online_tcp_listener(
  addr : @socket.Addr,
) -> @socket.TcpServer {
  TcpServer(addr, reuse_addr=true) catch {
    error if @async.is_being_cancelled() => raise error
    _ => raise luna_online_tcp_error(LunaOnlineTcpBind)
  }
}
