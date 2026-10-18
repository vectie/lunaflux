///|
/// External SIGTERM fixture: no GPU/model and no artificial delay in serving.
/// The bounded synchronous hold exposes the real prepare -> async bind edge.
async fn run_luna_openai_startup_stop_evidence(
  executable : Bytes,
  fixture_root : String,
) -> Unit {
  let limits = @online_tcp.LunaOnlineTcpLimits::new(
    read_chunk_bytes=2048,
    write_chunk_bytes=16384,
    accept_timeout_millis=1,
    input_idle_timeout_millis=1000,
    write_timeout_millis=1000,
    reactor_transition_budget=1,
  )
  let prepared = prepare_openai_pool_e2e(executable, limits, fixture_root~)
  println("startup-stop fixture worker prepared; synchronous hold begins")
  let started = @env.now()
  while @env.now() - started < 5000UL {
    ()
  }
  let pool = @online_tcp.bind_luna_online_openai_connection_pool(
    @socket.Addr::parse("127.0.0.1:0"),
    prepared,
    limits,
    log_capacity=64,
  )
  defer @async.protect_from_cancel(() => drain_openai_pool(pool))
  println("startup-stop fixture listener bound")
  while true {
    ignore(pool.progress_on_reactor())
    @async.pause()
  }
}
