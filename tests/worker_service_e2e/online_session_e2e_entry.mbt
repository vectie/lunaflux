///|
async fn main {
  match @env.args() {
    [_, "--framed-connection-pool", executable, fixture_root] =>
      try
        run_luna_framed_pool_evidence(@utf8.encode(executable), fixture_root)
      catch {
        @online_session.LunaOnlineInstancePreparationFailed(failure) => {
          println(
            "LunaFlux framed connection-pool preparation failure=\{Repr(failure)}",
          )
          raise @online_session.LunaOnlineInstancePreparationFailed(failure)
        }
        @online_tcp.LunaOnlineTcpFailed(rule) => {
          let rule_name = match rule {
            LunaOnlineTcpLimits => "limits"
            LunaOnlineTcpPreparation => "preparation"
            LunaOnlineTcpBind => "bind"
            LunaOnlineTcpAccept => "accept"
            LunaOnlineTcpRead => "read"
            LunaOnlineTcpWrite => "write"
            LunaOnlineTcpTimeout => "timeout"
            LunaOnlineTcpCancelled => "cancelled"
            LunaOnlineTcpCoordinator => "coordinator"
            LunaOnlineTcpTelemetry => "telemetry"
            LunaOnlineTcpLifecycle => "lifecycle"
            LunaOnlineTcpOwnership => "ownership"
          }
          println("LunaFlux framed connection-pool TCP rule=\{rule_name}")
          raise @online_tcp.LunaOnlineTcpFailed(rule)
        }
        error => {
          println("LunaFlux framed connection-pool failure=\{error}")
          raise error
        }
      } noraise {
        _ => println("LunaFlux framed connection-pool evidence passed")
      }
    [_, "--soak-24h", executable, fixture_root] =>
      try! run_phase23_soak_24h(@utf8.encode(executable), fixture_root)
    [_, "--soak-smoke", executable, fixture_root] =>
      try! run_phase23_soak_smoke(@utf8.encode(executable), fixture_root)
    [_, "--soak-cycle-fast", executable, fixture_root] =>
      try! run_phase23_soak_cycle_replay(
        @utf8.encode(executable),
        fixture_root,
        false,
      )
    [_, "--soak-cycle-timer", executable, fixture_root] =>
      try! run_phase23_soak_cycle_replay(
        @utf8.encode(executable),
        fixture_root,
        true,
      )
    [_, "--pipeline-early-cancel", executable] => {
      let cancelled = run_luna_pipeline_early_cancel_evidence(
        @utf8.encode(executable),
      ) catch {
        @online_tcp.LunaOnlineTcpFailed(rule) => {
          let rule_name = match rule {
            LunaOnlineTcpLimits => "limits"
            LunaOnlineTcpPreparation => "preparation"
            LunaOnlineTcpBind => "bind"
            LunaOnlineTcpAccept => "accept"
            LunaOnlineTcpRead => "read"
            LunaOnlineTcpWrite => "write"
            LunaOnlineTcpTimeout => "timeout"
            LunaOnlineTcpCancelled => "cancelled"
            LunaOnlineTcpCoordinator => "coordinator"
            LunaOnlineTcpTelemetry => "telemetry"
            LunaOnlineTcpLifecycle => "lifecycle"
            LunaOnlineTcpOwnership => "ownership"
          }
          println("LunaFlux pipeline early cancel TCP rule=\{rule_name}")
          raise @online_tcp.LunaOnlineTcpFailed(rule)
        }
        error => {
          println("LunaFlux pipeline early cancel failure=\{error}")
          raise error
        }
      }
      println(
        if cancelled {
          "LunaFlux pipeline early cancel disposition=accepted"
        } else {
          "LunaFlux pipeline early cancel disposition=typed-coordinator-rejection"
        },
      )
    }
    [_, "--openai-connection-pool", executable, fixture_root] =>
      try
        run_luna_openai_pool_evidence(@utf8.encode(executable), fixture_root~)
      catch {
        @online_tcp.LunaOnlineTcpFailed(rule) => {
          let rule_name = match rule {
            LunaOnlineTcpLimits => "limits"
            LunaOnlineTcpPreparation => "preparation"
            LunaOnlineTcpBind => "bind"
            LunaOnlineTcpAccept => "accept"
            LunaOnlineTcpRead => "read"
            LunaOnlineTcpWrite => "write"
            LunaOnlineTcpTimeout => "timeout"
            LunaOnlineTcpCancelled => "cancelled"
            LunaOnlineTcpCoordinator => "coordinator"
            LunaOnlineTcpTelemetry => "telemetry"
            LunaOnlineTcpLifecycle => "lifecycle"
            LunaOnlineTcpOwnership => "ownership"
          }
          println("LunaFlux OpenAI connection-pool TCP rule=\{rule_name}")
          raise @online_tcp.LunaOnlineTcpFailed(rule)
        }
        error => {
          println("LunaFlux OpenAI connection-pool failure=\{error}")
          raise error
        }
      } noraise {
        _ => println("LunaFlux OpenAI connection-pool evidence passed")
      }
    [_, "--openai-connection-pool", executable] =>
      try run_luna_openai_pool_evidence(@utf8.encode(executable)) catch {
        @online_tcp.LunaOnlineTcpFailed(rule) => {
          let rule_name = match rule {
            LunaOnlineTcpLimits => "limits"
            LunaOnlineTcpPreparation => "preparation"
            LunaOnlineTcpBind => "bind"
            LunaOnlineTcpAccept => "accept"
            LunaOnlineTcpRead => "read"
            LunaOnlineTcpWrite => "write"
            LunaOnlineTcpTimeout => "timeout"
            LunaOnlineTcpCancelled => "cancelled"
            LunaOnlineTcpCoordinator => "coordinator"
            LunaOnlineTcpTelemetry => "telemetry"
            LunaOnlineTcpLifecycle => "lifecycle"
            LunaOnlineTcpOwnership => "ownership"
          }
          println("LunaFlux OpenAI connection-pool TCP rule=\{rule_name}")
          raise @online_tcp.LunaOnlineTcpFailed(rule)
        }
        error => {
          println("LunaFlux OpenAI connection-pool failure=\{error}")
          raise error
        }
      } noraise {
        _ => println("LunaFlux OpenAI connection-pool evidence passed")
      }
    [_, "--operational-http", executable, .. roots] if roots.length() <= 1 =>
      try
        run_operational_http_only(
          @utf8.encode(executable),
          fixture_root=if roots.length() == 1 {
            roots[0]
          } else {
            "/private/tmp"
          },
        )
      catch {
        @online_tcp.LunaOnlineTcpFailed(rule) => {
          println("LunaFlux operational HTTP TCP rule=\{Repr(rule)}")
          raise @online_tcp.LunaOnlineTcpFailed(rule)
        }
        error => {
          println("LunaFlux operational HTTP failure=\{error}")
          raise error
        }
      } noraise {
        _ => println("LunaFlux operational HTTP evidence passed")
      }
    [_, "--balance-10000", executable, fixture_root] => {
      let mut balance_complete = false
      defer (if !balance_complete {
        println("LunaFlux Phase 2/3 balance failure")
      })
      run_phase23_balance_10000(@utf8.encode(executable), fixture_root)
      balance_complete = true
    }
    [_, executable, failure_executable, persistent_executable] =>
      try! run(
        @utf8.encode(executable),
        Some(@utf8.encode(failure_executable)),
        Some(@utf8.encode(persistent_executable)),
      )
    [_, executable, failure_executable] =>
      try! run(
        @utf8.encode(executable),
        Some(@utf8.encode(failure_executable)),
        None,
      )
    [_, executable] => try! run(@utf8.encode(executable), None, None)
    _ => println("worker executable path is required")
  }
}
