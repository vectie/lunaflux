///|
/// Owns the single native listener bind policy for LunaFlux servers.
/// `SO_REUSEADDR` is required for deterministic same-address restart after a
/// server actively closes an authenticated or rejected connection. This does
/// not enable `SO_REUSEPORT`; a concurrently live exact-address listener still
/// retains exclusive bind ownership.
async fn bind_luna_online_tcp_listener(
  addr : @socket.Addr,
) -> @socket.TcpServer {
  // Settle cancellation queued during synchronous model preparation before
  // starting the first asynchronous socket job. This is a startup checkpoint,
  // not a delay or an operation on the token-step path.
  @async.pause()
  TcpServer(addr, reuse_addr=true) catch {
    error if @async.is_being_cancelled() => raise error
    _ => raise luna_online_tcp_error(LunaOnlineTcpBind)
  }
}

///|
/// Claim the prepared worker before the cancellable listener boundary. Failed
/// binding/cancellation retains its sole cleanup obligation here; a successful
/// return transfers both owners together to the HTTP pool.
async fn bind_prepared_luna_openai_listener(
  addr : @socket.Addr,
  preparation : @online_session.LunaOnlineFramedServicePreparation,
) -> (@socket.TcpServer, @online_session.LunaOnlineFramedService) {
  let service = preparation.take_ready() catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let mut transferred = false
  defer (if !transferred {
    @async.protect_from_cancel(() => {
      service.begin_drain()
      while !service.is_closed() {
        ignore(service.progress())
        @async.pause()
      }
    })
  })
  let listener = bind_luna_online_tcp_listener(addr)
  transferred = true
  (listener, service)
}
