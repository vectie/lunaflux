///|
fn native_run_file_limits() -> @runtime_instance.RuntimeInstanceFileLimits raise {
  @runtime_instance.RuntimeInstanceFileLimits::new(
    launch=@launch_file.LaunchFileLimits::new(
      maximum_bytes=1048576L,
      maximum_json_depth=16,
    ),
    descriptor=@descriptor_file.RuntimeDescriptorFileLimits::new(
      max_descriptor_bytes=1048576L,
      max_json_depth=16,
    ),
    policy=@instance_policy_file.InstancePolicyFileLimits::new(
      max_policy_bytes=1048576L,
      max_json_depth=16,
    ),
    executable=@worker_executable_file.WorkerExecutableFileLimits::new(
      maximum_bytes=1073741824L,
    ),
  )
}

///|
/// Selects only the production opaque one-argument run grammar. Recipe and
/// family selection remain inside the authenticated launch file.
fn exact_opaque_run_argument(arguments : ArrayView[String]) -> String? {
  match arguments {
    [_, "run", model] => Some(model)
    _ => None
  }
}

///|
/// Selects only the production digest-pinned one-argument benchmark grammar.
fn exact_opaque_bench_argument(arguments : ArrayView[String]) -> String? {
  match arguments {
    [_, "bench", deployment] => Some(deployment)
    _ => None
  }
}

///|
/// Canonical inert operator diagnostics. Each variant retains the one opaque
/// deployment argument only until the shared release admission consumes it.
enum ReleaseDiagnosticArgument {
  ReleaseDoctor(String)
  ReleasePlan(String)
  ReleaseInspectKernels(String)
} derive(Debug, Eq)

///|
/// Selects only the three canonical digest-pinned diagnostic grammars. The
/// deployment argument is deliberately not interpreted here: the production
/// launch-file admission remains the single authority for absolute-label and
/// lowercase SHA-256 validation. Consequently, a malformed or unpinned
/// operand reaches that fail-closed admission and can never fall through to a
/// legacy model-root preflight.
fn exact_release_diagnostic_argument(
  arguments : ArrayView[String],
) -> ReleaseDiagnosticArgument? {
  match arguments {
    [_, "doctor", deployment] => Some(ReleaseDoctor(deployment))
    [_, "plan", deployment] => Some(ReleasePlan(deployment))
    [_, "inspect-kernels", deployment] =>
      Some(ReleaseInspectKernels(deployment))
    _ => None
  }
}

///|
fn release_diagnostic_header(command : ReleaseDiagnosticArgument) -> String {
  match command {
    ReleaseDoctor(_) => "LunaFlux authenticated deployment diagnostic"
    ReleasePlan(_) => "LunaFlux authenticated deployment plan"
    ReleaseInspectKernels(_) =>
      "LunaFlux authenticated deployment kernel inspection"
  }
}

///|
fn release_diagnostic_terminal_line(
  command : ReleaseDiagnosticArgument,
) -> String {
  match command {
    ReleaseDoctor(_) => "diagnostic report available: true"
    ReleasePlan(_) => "resolved deployment plan available: true"
    ReleaseInspectKernels(_) => "kernel release admission available: true"
  }
}

///|
/// Runs the same recipe-specific deployment admission used by `run` and
/// `bench`, then publishes only root-free evidence. `preflight_release` closes
/// every filesystem capability and never opens a device, spawns a process, or
/// binds a listener. No output is emitted before that admission succeeds.
fn run_release_diagnostic(command : ReleaseDiagnosticArgument) -> Unit raise {
  let deployment = match command {
    ReleaseDoctor(value) => value
    ReleasePlan(value) => value
    ReleaseInspectKernels(value) => value
  }
  let admitted = @runtime_instance.preflight_release(
    deployment,
    native_run_file_limits(),
  )
  println(release_diagnostic_header(command))
  println(admitted.evidence())
  println("release_preflight_sha256={admitted.evidence_sha256()}")
  println(release_diagnostic_terminal_line(command))
  println("filesystem authority retained: false")
  println("device opened: false")
  println("process spawned: false")
  println("listener bound: false")
  println("readiness: false")
}

///|
fn materialized_bundle_limits() -> @deployment_bundle.DeploymentBundleLimits raise {
  @deployment_bundle.DeploymentBundleLimits::new(
    maximum_bytes=1048576,
    maximum_base_image_bytes=4096,
    maximum_locator_bytes=4096,
  )
}

///|
/// Publishes only root-free offline evidence after the production semantic join
/// has consumed every filesystem authority. No device or listener is opened.
fn validate_release(deployment : String) -> Unit raise {
  let admitted = @runtime_instance.preflight_release(
    deployment,
    native_run_file_limits(),
  )
  println(admitted.evidence())
  println("release_preflight_sha256={admitted.evidence_sha256()}")
}

///|
/// Validates a claimed no-overwrite bundle through typed staged-source to
/// final-target bindings and publishes only root-free semantic evidence.
fn validate_materialized_release(bundle : String) -> Unit raise {
  let admitted = @runtime_instance.preflight_materialized_release(
    bundle,
    native_run_file_limits(),
    materialized_bundle_limits(),
  )
  println(admitted.evidence())
  println(
    "release_materialization_preflight_sha256={admitted.evidence_sha256()}",
  )
}

///|
enum NativeDrainAction {
  NativeDrainComplete
  NativeDrainRequest
  NativeDrainProgress
} derive(Debug, Eq)

///|
fn native_drain_action(
  phase : @runtime_instance.RuntimeInstancePhase,
  cleanup_complete : Bool,
) -> NativeDrainAction {
  if cleanup_complete {
    return NativeDrainComplete
  }
  match phase {
    Starting | HealthyNotReady | Ready | OpenAiQualificationReady =>
      NativeDrainRequest
    Draining | Closed | Failed => NativeDrainProgress
  }
}

///|
/// Stable, payload-safe operator projection shared by the opaque CLI tests and
/// the production Ready publication path.
fn startup_measurement_lines(
  cold_start_latency_millis : Int,
  readiness : @runtime_instance.RuntimeInstanceReadiness,
) -> (String, String) {
  (
    "cold_start_latency_millis: " + cold_start_latency_millis.to_string(),
    "startup_measurement_readiness: " +
    (match readiness {
      RuntimeReady => "true"
      RuntimeNotReady => "false"
    }),
  )
}

///|
fn runtime_origin_line(origin : String) -> String {
  "runtime_origin=" + origin
}

///|
fn control_origin_line(origin : String) -> String {
  "control_origin=" + origin
}

///|
/// Drives deterministic bounded cleanup without abandoning the owner.
async fn drain_native_owner(
  owner : @runtime_instance.RuntimeInstanceOwner,
) -> Bool {
  @async.protect_from_cancel(() => {
    for cleanup_step = 0
        cleanup_step < 1048576
        cleanup_step = cleanup_step + 1 {
      match native_drain_action(owner.phase(), owner.cleanup_complete()) {
        NativeDrainComplete => return true
        NativeDrainRequest => owner.begin_drain() catch { _ => () }
        NativeDrainProgress => ()
      }
      ignore(owner.progress())
    }
    owner.cleanup_complete()
  })
}

///|
/// Consumes optional inherited FD 6 before any model, device, worker, or
/// listener authority is constructed. A present descriptor must provide one
/// exact bounded frame and write-side EOF; malformed input never falls back to
/// native service activation.
async fn acquire_opaque_inference_credential() -> @inference_credential.InheritedInferenceAuthPolicy? {
  let optional = @inference_credential.prepare_optional_inherited_inference_credential_v1()
  guard optional is Some(owner) else { return None }
  @async.protect_from_cancel(() => {
    for transition = 0; transition < 1048576; transition = transition + 1 {
      match owner.progress() {
        InheritedInferenceCredentialReady =>
          return Some(
            owner.take_startup_auth_policy_v1(maximum_credential_bytes=4096),
          )
        InheritedInferenceCredentialPending => @async.pause()
        InheritedInferenceCredentialFailedProgress =>
          return Some(
            owner.take_startup_auth_policy_v1(maximum_credential_bytes=4096),
          )
      }
    }
    owner.close() catch {
      _ => ()
    }
    raise @inference_credential.InheritedInferenceCredentialFailed(
      CredentialChannelLost,
    )
  })
}

///|
/// Consumes optional inherited FD 7 before any model, device, worker, or
/// listener authority exists. The returned capability can authenticate one
/// exact launch/manifest claim and exposes no verifier-key bytes.
async fn acquire_luna_promotion_verifier() -> @luna_capability_manifest.LunaStartupExternalApprovalHandoff? {
  let optional = @luna_capability_manifest.prepare_optional_inherited_luna_startup_verifier_key_v1()
  guard optional is Some(owner) else { return None }
  @async.protect_from_cancel(() => {
    for transition = 0; transition < 1048576; transition = transition + 1 {
      match owner.progress() {
        LunaStartupVerifierKeyReady => return Some(owner.take_handoff())
        LunaStartupVerifierKeyPending => @async.pause()
        LunaStartupVerifierKeyFailedProgress =>
          return Some(owner.take_handoff())
      }
    }
    owner.close() catch {
      _ => ()
    }
    raise @luna_capability_manifest.Invalid(ExternalApproval)
  })
}

///|
fn close_untransferred_startup_inputs(
  credential : @inference_credential.InheritedInferenceAuthPolicy?,
  promotion : @luna_capability_manifest.LunaStartupExternalApprovalHandoff?,
) -> Unit {
  if credential is Some(owner) {
    owner.close()
  }
  if promotion is Some(owner) {
    owner.close()
  }
}

///|
fn[T] close_drain_after_runtime_prepare_failure(
  drain : @inherited_drain.InheritedDrainPreparation,
  error : Error,
) -> T raise {
  drain.close() catch {
    _ => ()
  }
  raise error
}

///|
async fn[T] close_runtime_after_drain_attach_failure(
  owner : @runtime_instance.RuntimeInstanceOwner,
  drain : @inherited_drain.InheritedDrainPreparation,
  error : Error,
) -> T {
  drain.close() catch {
    _ => ()
  }
  ignore(drain_native_owner(owner))
  raise error
}

///|
/// Drives bounded lower transitions; cancellation always closes before exit.
async fn run_native_model(model : String) -> Unit {
  let credential = acquire_opaque_inference_credential()
  let promotion_result : Result[
    @luna_capability_manifest.LunaStartupExternalApprovalHandoff?,
    Error,
  ] = Ok(acquire_luna_promotion_verifier()) catch { error => Err(error) }
  let promotion = match promotion_result {
    Err(error) => {
      close_untransferred_startup_inputs(credential, None)
      raise error
    }
    Ok(value) => value
  }
  // The opaque process path requires possession of the deployment-created
  // inherited FD 5 capability before model/device/listener construction.
  let drain_result : Result[@inherited_drain.InheritedDrainPreparation, Error] = Ok(
    @inherited_drain.prepare_inherited_drain_v1(),
  ) catch {
    error => Err(error)
  }
  let drain = match drain_result {
    Err(error) => {
      close_untransferred_startup_inputs(credential, promotion)
      raise error
    }
    Ok(value) => value
  }
  let owner = @runtime_instance.prepare_opaque_cli(
    model,
    native_run_file_limits(),
    credential,
    promotion_verifier=promotion,
  ) catch {
    error => {
      close_untransferred_startup_inputs(credential, promotion)
      close_drain_after_runtime_prepare_failure(drain, error)
    }
  }
  owner.attach_inherited_drain_v1(drain) catch {
    error => close_runtime_after_drain_attach_failure(owner, drain, error)
  }
  let mut ready_printed = false
  for transitions = 0; transitions < 1048576; transitions = transitions + 1 {
    if @async.is_being_cancelled() {
      let closed = drain_native_owner(owner)
      if !closed {
        raise @runtime_instance.RuntimeInstanceFailedAndCleanup(
          Lifecycle,
          Cleanup,
        )
      }
      // Re-observe cancellation only after deterministic owner cleanup.
      @async.pause()
      return
    }
    ignore(owner.progress())
    match owner.phase() {
      Ready =>
        if !ready_printed {
          match owner.service_protocol() {
            RuntimeNativeFramedV1 => println("LunaFlux native framed service")
            RuntimeOpenAiResponsesV1 =>
              println("LunaFlux OpenAI Responses service")
          }
          println("health: healthy")
          println("readiness: true")
          let endpoint = owner.bound_endpoint()
          println(runtime_origin_line(endpoint.origin()))
          println(control_origin_line(owner.control_origin()))
          match endpoint.protocol() {
            RuntimeNativeFramedV1 =>
              println("runtime_protocol=native-framed-v1")
            RuntimeOpenAiResponsesV1 =>
              println("runtime_protocol=openai-responses-v1")
          }
          let startup = owner.startup_measurement()
          let (latency_line, readiness_line) = startup_measurement_lines(
            startup.cold_start_latency_millis(),
            startup.readiness(),
          )
          println(latency_line)
          println(readiness_line)
          ready_printed = true
        }
      Closed => return
      Failed if owner.cleanup_complete() =>
        raise @runtime_instance.RuntimeInstanceFailed(Lifecycle)
      _ => ()
    }
  }
  if drain_native_owner(owner) {
    raise @runtime_instance.RuntimeInstanceFailed(Lifecycle)
  }
  raise @runtime_instance.RuntimeInstanceFailedAndCleanup(Lifecycle, Cleanup)
}

///|
/// Publishes exact raw local measurement evidence only after the benchmark
/// owner has drained and closed every runtime resource.
async fn run_native_benchmark(deployment : String) -> Unit {
  let evidence = @runtime_instance.benchmark_local_deployment(
    deployment,
    native_run_file_limits(),
  )
  println("LunaFlux bounded local benchmark")
  println(@utf8.decode(evidence.canonical_bytes()))
  println("local_benchmark_sha256=" + evidence.digest().as_hex())
  println("comparison admitted: false")
  println("promotion authorized: false")
}
