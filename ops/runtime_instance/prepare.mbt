///|
/// Prepares the sole existing scheduler/service/worker composition.
pub fn prepare(
  model_argument : StringView,
  limits : RuntimeInstanceFileLimits,
) -> RuntimeInstanceOwner raise RuntimeInstanceError {
  prepare_with_startup_credential(
    model_argument,
    limits,
    None,
    None,
    opaque_cli_requires_drain=false,
  )
}

///|
/// Prepares the opaque one-argument runtime from authenticated policy plus an
/// optional already-consumed inherited credential. Native v1/v2 accepts no
/// credential; OpenAI policy v3 requires exactly one.
pub fn prepare_opaque_cli(
  model_argument : StringView,
  limits : RuntimeInstanceFileLimits,
  credential : @inference_credential.InheritedInferenceAuthPolicy?,
  promotion_verifier? : @luna_capability_manifest.LunaStartupExternalApprovalHandoff? = None,
) -> RuntimeInstanceOwner raise RuntimeInstanceError {
  prepare_with_startup_credential(
    model_argument,
    limits,
    credential,
    promotion_verifier,
    opaque_cli_requires_drain=true,
  )
}

///|
fn prepare_with_startup_credential(
  model_argument : StringView,
  limits : RuntimeInstanceFileLimits,
  credential : @inference_credential.InheritedInferenceAuthPolicy?,
  promotion_verifier : @luna_capability_manifest.LunaStartupExternalApprovalHandoff?,
  opaque_cli_requires_drain~ : Bool,
) -> RuntimeInstanceOwner raise RuntimeInstanceError {
  let clock = @monotonic_clock.MonotonicClock::system()
  let started_at_millis = clock.now_millis() catch { _ => raise fail(Clock) }
  let admitted = admit_release_inputs(
    model_argument,
    limits,
    luna_verifier=promotion_verifier,
  )
  let binding_result : Result[RuntimeOpaqueStartupBinding, RuntimeInstanceError] = Ok(
    select_opaque_startup_binding(admitted.instance, credential),
  ) catch {
    error => Err(error)
  }
  let startup_binding = match binding_result {
    Ok(value) => value
    Err(error) => {
      if credential is Some(inherited) {
        inherited.close()
      }
      close_model_kernel(admitted.model_root, admitted.kernel_root)
      raise error
    }
  }
  let physical_result : Result[Unit, RuntimeInstanceError] = Ok(
    admitted.runtime.admit_physical(),
  ) catch {
    error => Err(error)
  }
  match physical_result {
    Ok(_) => ()
    Err(error) => {
      startup_binding.close_authentication()
      close_model_kernel(admitted.model_root, admitted.kernel_root)
      raise error
    }
  }
  let executable_result : Result[
    @worker_executable_file.WorkerExecutableAdmission,
    RuntimeInstanceError,
  ] = Ok(verify_release_worker(admitted.launch, limits.executable)) catch {
    error => Err(error)
  }
  let executable = match executable_result {
    Ok(value) => value
    Err(error) => {
      startup_binding.close_authentication()
      close_model_kernel(admitted.model_root, admitted.kernel_root)
      raise error
    }
  }
  let release_result : Result[RuntimeReleaseAdmission, RuntimeInstanceError] = Ok(
    build_release_admission(admitted, executable),
  ) catch {
    error => Err(error)
  }
  let release_admission = match release_result {
    Ok(value) => value
    Err(error) => {
      executable.close() catch {
        _ => ()
      }
      startup_binding.close_authentication()
      close_model_kernel(admitted.model_root, admitted.kernel_root)
      raise error
    }
  }
  let envelope_result : Result[RuntimeServiceEnvelope, RuntimeInstanceError] = Ok(
    admitted.instance.service_envelope(),
  ) catch {
    error => Err(error)
  }
  let envelope = match envelope_result {
    Ok(value) => value
    Err(error) => {
      executable.close() catch {
        _ => ()
      }
      startup_binding.close_authentication()
      close_model_kernel(admitted.model_root, admitted.kernel_root)
      raise error
    }
  }
  let service = admitted.instance.prepare_owned_service(
    executable,
    admitted.model_root,
    admitted.kernel_root,
  ) catch {
    _ => {
      // The lower contract raises only before private root-pair acquisition.
      executable.close() catch {
        _ => ()
      }
      startup_binding.close_authentication()
      close_model_kernel(admitted.model_root, admitted.kernel_root)
      raise fail(ServicePreparation)
    }
  }
  capture_prepared_owner(
    service,
    admitted.model_root,
    admitted.kernel_root,
    envelope,
    release_admission,
    startup_binding,
    opaque_cli_requires_drain,
    clock,
    started_at_millis,
  )
}
