///|
fn output_path_unit_is_safe(unit : Char) -> Bool {
  unit is ('A'..='Z' | 'a'..='z' | '0'..='9' | '.' | '_' | '-' | '/')
}

///|
fn validate_new_output_path(
  path : String,
) -> (String, String) raise LunaBf16ReleaseBindError {
  if path.length() < 2 ||
    path.length() > 4096 ||
    path[0] != '/' ||
    path[path.length() - 1] == '/' ||
    path.contains("//") ||
    path.contains("/./") ||
    path.contains("/../") ||
    path.has_suffix("/.") ||
    path.has_suffix("/..") {
    raise Publication
  }
  for unit in path {
    if !output_path_unit_is_safe(unit) {
      raise Publication
    }
  }
  guard path.rev_find("/") is Some(separator) && separator < path.length() - 1 else {
    raise Publication
  }
  let parent = if separator == 0 { "/" } else { path[:separator].to_owned() }
  (parent, path[separator + 1:].to_owned())
}

///|
async fn staging_path(path : String) -> String raise LunaBf16ReleaseBindError {
  let (parent, basename) = validate_new_output_path(path)
  let canonical = @fs.realpath(parent) catch {
    error => {
      ignore(error)
      raise Publication
    }
  }
  let output_exists = @fs.exists(path) catch {
    error => {
      ignore(error)
      raise Publication
    }
  }
  if canonical != parent || output_exists {
    raise Publication
  }
  let staging = if parent == "/" {
    "/." + basename + ".lunaflux-release-bind"
  } else {
    parent + "/." + basename + ".lunaflux-release-bind"
  }
  let staging_exists = @fs.exists(staging) catch {
    error => {
      ignore(error)
      raise Publication
    }
  }
  if staging_exists {
    raise Publication
  }
  staging
}

///|
async fn write_new(
  path : String,
  bytes : Bytes,
) -> Unit raise LunaBf16ReleaseBindError {
  @fs.write_file(
    path,
    bytes,
    create_mode=CreateNew,
    permission=0o400,
    sync=Full,
  ) catch {
    error => {
      ignore(error)
      raise Publication
    }
  }
}

///|
async fn remove_staging(path : String) -> Bool noraise {
  try @fs.rmdir(path, recursive=true) catch {
    error => {
      ignore(error)
      false
    }
  } noraise {
    _ => true
  }
}

///|
fn step_limits() -> @device_step.DeviceStepLimits raise LunaBf16ReleaseBindError {
  @device_step.DeviceStepLimits::new(
    max_rows=1,
    max_tokens=1,
    max_pages=RELEASE_MAX_PAGE_TABLE_ENTRIES,
    max_sequence_tokens=RELEASE_MAX_SEQUENCE_TOKENS,
    max_token_id=2999,
  ) catch {
    error => {
      ignore(error)
      raise KernelBinding
    }
  }
}

///|
fn execution_limits(
  operation_count : Int,
) -> @execution_manifest_file.ExecutionManifestFileLimits raise LunaBf16ReleaseBindError {
  @execution_manifest_file.ExecutionManifestFileLimits::new(
    max_manifest_bytes=1048576L,
    max_json_depth=16,
    max_modules=operation_count,
    max_operations=operation_count,
    device_memory=@device_memory.DeviceMemoryLimits::new(
      activation_alignment=256L,
      max_arena_bytes=33554432L,
    ),
    launch_contract=@launch_contract.LaunchContractLimits::new(
      max_profiles=1,
      max_contracts=operation_count,
      max_operands_per_contract=32,
    ),
    artifact_file=@artifact_file.KernelArtifactFileLimits::new(
      max_manifest_bytes=1048576L,
      max_json_depth=16,
      max_path_bytes=4096,
      artifact=@artifact.KernelArtifactLimits::new(
        max_modules=operation_count,
        max_entry_points=operation_count,
        max_module_bytes=4194304L,
        max_total_module_bytes=67108864L,
        max_function_symbol_bytes=1024,
      ),
    ),
  ) catch {
    error => {
      ignore(error)
      raise KernelBinding
    }
  }
}

///|
fn derive_bootstrap(
  staging : String,
  inputs : ReleaseBindInputs,
  layout : @device_layout.DeviceKvLayout,
  target : @catalog.DeviceTarget,
  root_plan : @kernel_root.LunaKernelRootPlan,
) -> String raise LunaBf16ReleaseBindError {
  let root = @approved_fs.ApprovedRoot::open_absolute(staging + "/payload") catch {
    error => {
      ignore(error)
      raise KernelBinding
    }
  }
  let result : Result[String, LunaBf16ReleaseBindError] = Ok(
    {
      let operation_count = inputs.plan.operations().length()
      let execution = @execution_manifest_file.load_paged(
        root,
        @approved_fs.ApprovedRelativeLocator::new(root_plan.manifest_relative()),
        @execution_manifest_file.ExecutionManifestDigest::from_sha256(
          root_plan.manifest_sha256(),
        ),
        inputs.plan,
        inputs.weights.layout(),
        layout,
        target,
        step_limits(),
        execution_limits(operation_count),
      )
      @device_step.admit_device_worker_bootstrap(
        execution.blueprint(),
        execution.artifacts(),
        device_ordinal=0,
        @device_step.DeviceBootstrapLimits::new(
          max_modules=operation_count,
          max_entry_points=operation_count,
          max_steps=operation_count,
          max_operands_per_step=32,
          max_symbol_bytes=1024,
          max_canonical_bytes=16777216,
        ),
      )
      .digest()
      .as_hex()
    },
  ) catch {
    error => {
      ignore(error)
      Err(KernelBinding)
    }
  }
  let close_result : Result[Unit, LunaBf16ReleaseBindError] = Ok(root.close()) catch {
    error => {
      ignore(error)
      Err(KernelBinding)
    }
  }
  match (result, close_result) {
    (Ok(value), Ok(_)) => value
    _ => raise KernelBinding
  }
}

///|
async fn publish_release_binding(
  output : String,
  inputs : ReleaseBindInputs,
  release : @luna_bf16_kernel_producer.LunaBf16KernelReleaseInput,
) -> ReleaseBindEvidence raise LunaBf16ReleaseBindError {
  let staging = staging_path(output)
  @fs.mkdir(staging, permission=0o700) catch {
    error => {
      ignore(error)
      raise Publication
    }
  }
  let result : Result[ReleaseBindEvidence, LunaBf16ReleaseBindError] = Ok(
    {
      @fs.mkdir(staging + "/payload", permission=0o700) catch {
        error => {
          ignore(error)
          raise Publication
        }
      }
      let root_plan = @kernel_root.plan_luna_bf16_kernel_root(
        release,
        @kernel_root.LunaKernelRootLimits::new(
          max_files=33,
          max_inventory_bytes=65536,
          max_evidence_bytes=65536,
        ),
      ) catch {
        error => {
          ignore(error)
          raise KernelBinding
        }
      }
      for file in root_plan.files() {
        let relative = file.relative_path()
        if relative.has_prefix("sha256/") {
          let directory = staging + "/payload/sha256"
          let exists = @fs.exists(directory) catch {
            error => {
              ignore(error)
              raise Publication
            }
          }
          if !exists {
            @fs.mkdir(directory, permission=0o700) catch {
              error => {
                ignore(error)
                raise Publication
              }
            }
          }
        }
        write_new(staging + "/payload/" + relative, file.bytes())
      }
      write_new(staging + "/kernel.files.sha256", root_plan.inventory_bytes())
      write_new(staging + "/kernel-root.plan.v1", root_plan.evidence_bytes())
      let bootstrap = derive_bootstrap(
        staging,
        inputs,
        release_layout(inputs.plan),
        release_target(),
        root_plan,
      )
      @fs.rename(staging, output, replace=false) catch {
        error => {
          ignore(error)
          raise Publication
        }
      }
      {
        model_plan_sha256: inputs.plan.identity().plan().as_hex(),
        manifest_sha256: root_plan.manifest_sha256(),
        bootstrap_sha256: bootstrap,
        inventory_sha256: root_plan.inventory_sha256(),
        plan_sha256: root_plan.evidence_sha256(),
      }
    },
  ) catch {
    error => {
      ignore(error)
      Err(Publication)
    }
  }
  match result {
    Ok(value) => value
    Err(error) => {
      if remove_staging(staging) {
        raise error
      }
      raise Publication
    }
  }
}
