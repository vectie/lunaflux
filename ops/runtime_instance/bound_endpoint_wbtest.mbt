///|
test "bound endpoint publication requires live readiness" {
  assert_true(
    runtime_bound_endpoint_publication_allowed(
      Ready,
      RuntimeHealthy,
      RuntimeReady,
    ),
  )
  for
    phase in [
      Starting,
      HealthyNotReady,
      OpenAiQualificationReady,
      Draining,
      Closed,
      Failed,
    ] {
    assert_false(
      runtime_bound_endpoint_publication_allowed(
        phase,
        RuntimeHealthy,
        RuntimeReady,
      ),
    )
  }
  assert_false(
    runtime_bound_endpoint_publication_allowed(
      Ready,
      RuntimeUnhealthy,
      RuntimeReady,
    ),
  )
  assert_false(
    runtime_bound_endpoint_publication_allowed(
      Ready,
      RuntimeHealthy,
      RuntimeNotReady,
    ),
  )
}

///|
test "owner never publishes an endpoint before readiness or during drain" {
  for phase in [Starting, HealthyNotReady, Draining, Closed, Failed] {
    let owner = empty_owner_for_state_test(phase)
    try owner.bound_endpoint() catch {
      RuntimeInstanceFailed(Lifecycle) => ()
      _ => abort("unexpected endpoint lifecycle rejection")
    } noraise {
      _ => abort("non-ready owner published a runtime origin")
    }
  }
}

///|
test "control origin publication requires one live healthy observational listener" {
  let live_states : FixedArray[@online_tcp.LunaOnlineHttpControlState] = [
    LunaOnlineHttpControlListening,
    LunaOnlineHttpControlConnected,
  ]
  for state in live_states {
    assert_true(
      runtime_control_origin_publication_allowed(
        Ready,
        RuntimeHealthy,
        1,
        state,
        LunaOnlineTcpServerHealthy,
      ),
    )
  }
  assert_eq(
    runtime_control_origin_from_addr(@socket.Addr::parse("127.0.0.1:43129")),
    "http://127.0.0.1:43129",
  )
  for
    phase in [
      Starting,
      HealthyNotReady,
      OpenAiQualificationReady,
      Draining,
      Closed,
      Failed,
    ] {
    assert_false(
      runtime_control_origin_publication_allowed(
        phase,
        RuntimeHealthy,
        1,
        LunaOnlineHttpControlListening,
        LunaOnlineTcpServerHealthy,
      ),
    )
    let owner = empty_owner_for_state_test(phase)
    try owner.control_origin() catch {
      RuntimeInstanceFailed(Lifecycle) => ()
      _ => abort("unexpected control-origin lifecycle rejection")
    } noraise {
      _ => abort("non-ready owner published a control origin")
    }
  }
  assert_false(
    runtime_control_origin_publication_allowed(
      Ready,
      RuntimeHealthy,
      1,
      LunaOnlineHttpControlDraining,
      LunaOnlineTcpServerHealthy,
    ),
  )
  assert_false(
    runtime_control_origin_publication_allowed(
      Ready,
      RuntimeHealthy,
      0,
      LunaOnlineHttpControlListening,
      LunaOnlineTcpServerHealthy,
    ),
  )
  assert_false(
    runtime_control_origin_publication_allowed(
      Ready,
      RuntimeHealthy,
      1,
      LunaOnlineHttpControlListening,
      LunaOnlineTcpServerFailed,
    ),
  )
}

///|
test "bound endpoint is root-free and rejects an unbound port" {
  let endpoint = runtime_bound_endpoint_from_addr(
    RuntimeOpenAiResponsesV1,
    @socket.Addr::parse("127.0.0.1:43127"),
  )
  assert_eq(endpoint.protocol(), RuntimeOpenAiResponsesV1)
  assert_eq(endpoint.origin(), "http://127.0.0.1:43127")
  let native = runtime_bound_endpoint_from_addr(
    RuntimeNativeFramedV1,
    @socket.Addr::parse("127.0.0.1:43128"),
  )
  assert_eq(native.origin(), "tcp://127.0.0.1:43128")
  try
    runtime_bound_endpoint_from_addr(
      RuntimeNativeFramedV1,
      @socket.Addr::parse("127.0.0.1:0"),
    )
  catch {
    RuntimeInstanceFailed(Lifecycle) => ()
    _ => abort("unexpected unbound endpoint rejection")
  } noraise {
    _ => abort("unbound endpoint acquired publication authority")
  }
}

///|
async test "listener collision fails closed and close permits exact rebind" {
  let first = @socket.TcpServer(
    @socket.Addr::parse("127.0.0.1:0"),
    reuse_addr=false,
  )
  let exact = first.addr
  try @socket.TcpServer(exact, reuse_addr=false) catch {
    _ => ()
  } noraise {
    duplicate => {
      duplicate.close()
      first.close()
      abort("second listener acquired an occupied runtime origin")
    }
  }
  first.close()
  let rebound = @socket.TcpServer(exact, reuse_addr=false)
  assert_eq(rebound.addr, exact)
  rebound.close()
}
