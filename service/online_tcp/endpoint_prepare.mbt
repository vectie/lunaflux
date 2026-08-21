///|
pub async fn bind_luna_online_tcp_single_endpoint(
  addr : @socket.Addr,
  preparation : @online_session.LunaOnlineFramedCoordinatorPreparation,
  limits : LunaOnlineTcpLimits,
) -> LunaOnlineTcpEndpoint {
  if !limits.is_valid() {
    raise luna_online_tcp_error(LunaOnlineTcpLimits)
  }
  let maximum_wait = preparation.maximum_transport_wait_millis() catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  if !limits.fits_transport_wait(maximum_wait) {
    raise luna_online_tcp_error(LunaOnlineTcpLimits)
  }
  let input = FixedArray::make(limits.read_chunk_bytes, b'\x00')
  let output = LunaOnlineTcpOutputScratch::new(limits.write_chunk_bytes) catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpLimits)
  }
  let servers : Array[@socket.TcpServer] = Array::new(capacity=1)
  let connections : Array[@socket.Tcp] = Array::new(capacity=1)
  let coordinators : Array[@online_session.LunaOnlineFramedCoordinator] = Array::new(
    capacity=1,
  )
  let offers : Array[@online_session.LunaOnlineFramedEventOffer] = Array::new(
    capacity=1,
  )
  let flights : Array[LunaOnlineTcpOutputFlight] = Array::new(capacity=1)
  let endpoint : LunaOnlineTcpEndpoint = {
    limits,
    bound_addr: addr,
    servers,
    connections,
    coordinators,
    input,
    output,
    offers,
    flights,
    activity: { reactor_active: false, maintenance_active: false },
    phase: LUNA_ONLINE_TCP_LISTENING,
    tail_start: 0,
    tail_length: 0,
    tail_backpressured: false,
    maintenance_handoff: { pending: false },
    close_pending: false,
    drain_requested: false,
    disconnect_requested: false,
    failure_rule: 0,
    rejection_sequence: 0UL,
    rejection_rule: 0,
    rejection_latched: false,
  }
  let server = @socket.TcpServer(addr) catch {
    error if @async.is_being_cancelled() => raise error
    _ => raise luna_online_tcp_error(LunaOnlineTcpBind)
  }
  if @async.is_being_cancelled() {
    server.close()
    @async.pause()
    raise luna_online_tcp_error(LunaOnlineTcpCancelled)
  }
  endpoint.bound_addr = server.addr
  endpoint.servers.push(server)
  let coordinator = preparation.take_ready() catch {
    _ => {
      server.close()
      endpoint.servers.clear()
      raise luna_online_tcp_error(LunaOnlineTcpPreparation)
    }
  }
  endpoint.coordinators.push(coordinator)
  endpoint
}
