///|
fn close_cleanup(
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
) -> LunaModelStartupCleanup? {
  let kernel_failed = Ref(false)
  let model_failed = Ref(false)
  kernel_root.close() catch {
    _ => kernel_failed.val = true
  }
  model_root.close() catch {
    _ => model_failed.val = true
  }
  match (model_failed.val, kernel_failed.val) {
    (false, false) => None
    (true, false) => Some(ModelRoot)
    (false, true) => Some(KernelRoot)
    (true, true) => Some(BothRoots)
  }
}

///|
fn close_after_kernel_open_failure(
  model_root : @approved_fs.ApprovedRoot,
) -> Unit raise LunaModelStartupError {
  model_root.close() catch {
    _ => raise FailedAndClose(primary=KernelRootAdmission, cleanup=ModelRoot)
  }
}

///|
fn descriptor_stage(
  _error : @descriptor_file.RuntimeDescriptorFileError,
) -> LunaModelStartupStage {
  RuntimeDescriptorAdmission
}

///|
fn probe_admitted(
  admission : @descriptor_file.RuntimeDescriptorAdmission,
) -> LunaPinnedModelPreflight {
  match @hardware.probe() {
    Unavailable(reason) => {
      let outcome = physical_gate(reason)
      { admission: Some(admission), outcome }
    }
    Available(driver_version=_, devices~) => {
      let mut selected = None
      for device in devices {
        if device.ordinal() == admission.device_ordinal() {
          selected = Some(device)
        }
      }
      match selected {
        None => {
          let outcome = LunaModelPhysicalGate(AssignedDeviceMissing)
          { admission: Some(admission), outcome }
        }
        Some(device) =>
          if !admission.target().matches_capability(device.capability()) {
            let outcome = LunaModelPhysicalGate(DeviceTargetMismatch)
            { admission: Some(admission), outcome }
          } else {
            { admission: Some(admission), outcome: LunaModelPreflightComplete }
          }
      }
    }
  }
}

///|
/// Opens two independent approved authorities, reconstructs every existing
/// inert model/kernel/bootstrap input, closes both roots in reverse acquisition
/// order, and only then checks the exact selected physical device.
pub fn preflight_pinned(
  model_root_label : String,
  kernel_root_label : String,
  descriptor_relative : String,
  descriptor_sha256 : String,
) -> LunaPinnedModelPreflight raise LunaModelStartupError {
  let locator = @approved_fs.ApprovedRelativeLocator::new(descriptor_relative) catch {
    _ => raise Failed(DescriptorLocatorAdmission)
  }
  let digest = @descriptor_file.RuntimeDescriptorDigest::from_sha256(
    descriptor_sha256,
  ) catch {
    _ => raise Failed(DescriptorDigestAdmission)
  }
  let limits = @descriptor_file.RuntimeDescriptorFileLimits::new(
    max_descriptor_bytes=1048576L,
    max_json_depth=16,
  ) catch {
    _ => raise Failed(DescriptorLimitAdmission)
  }
  let model_root = @approved_fs.ApprovedRoot::open_absolute(model_root_label) catch {
    _ => raise Failed(ModelRootAdmission)
  }
  let kernel_root = @approved_fs.ApprovedRoot::open_absolute(kernel_root_label) catch {
    _ => {
      close_after_kernel_open_failure(model_root)
      raise Failed(KernelRootAdmission)
    }
  }
  let loaded : Result[
    @descriptor_file.RuntimeDescriptorAdmission,
    LunaModelStartupStage,
  ] = Ok(
    @descriptor_file.load(
      model_root, kernel_root, model_root_label, kernel_root_label, locator, digest,
      limits,
    ),
  ) catch {
    error => Err(descriptor_stage(error))
  }
  let cleanup = close_cleanup(model_root, kernel_root)
  match (loaded, cleanup) {
    (Err(primary), Some(cleanup)) => raise FailedAndClose(primary~, cleanup~)
    (Err(primary), None) => raise Failed(primary)
    (Ok(_), Some(cleanup)) => raise CloseFailed(cleanup)
    (Ok(admission), None) => probe_admitted(admission)
  }
}
