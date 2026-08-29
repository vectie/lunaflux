///|
fn export_sha256(bytes : Bytes) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn export_target_name(
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
) -> String raise LunaBf16CandidateExportError {
  let target = set.target()
  if target.compute_major() <= 0 ||
    target.compute_minor() < 0 ||
    target.compute_minor() > 9 {
    raise ExportFailed(CandidateSet)
  }
  "sm_" +
  target.compute_major().to_string() +
  target.compute_minor().to_string()
}

///|
fn export_record(
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
) -> Bytes raise LunaBf16CandidateExportError {
  let identity = set.model_identity()
  let profile = set.profile()
  let compiler = set.compiler_policy()
  let compiler_identity = compiler.identity()
  @utf8.encode(
    "schema=lunaflux-bf16-candidate-export.v1\n" +
    "model_content_sha256=" +
    identity.content().as_hex() +
    "\nmodel_plan_sha256=" +
    identity.plan().as_hex() +
    "\ntarget=" +
    export_target_name(set) +
    "\nprofile_id=" +
    profile.id().as_int().to_string() +
    "\nmax_query_rows=" +
    profile.max_query_rows().to_string() +
    "\nmax_query_tokens=" +
    profile.max_query_tokens().to_string() +
    "\nmax_page_table_entries=" +
    profile.max_page_table_entries().to_string() +
    "\ntoolchain_sha256=" +
    compiler_identity.toolchain_digest_sha256() +
    "\ncompiler_version=" +
    compiler_identity.major().to_string() +
    "." +
    compiler_identity.minor().to_string() +
    "." +
    compiler_identity.patch().to_string() +
    "\ncandidate_set_sha256=" +
    set.candidate_set_sha256() +
    "\ncandidate_inventory_sha256=" +
    set.inventory_sha256() +
    "\noperation_count=" +
    set.operation_count().to_string() +
    "\ncandidate_root=candidate-root\n" +
    "candidate_inventory=candidate.files.sha256\n" +
    "generated_lowerings=1\ncaller_source_authority=0\n" +
    "compiler_invoked=0\ndevice_opened=0\nruntime_authority=0\n",
  )
}

///|
async fn write_new_file(
  path : String,
  bytes : Bytes,
) -> Unit raise LunaBf16CandidateExportError {
  @fs.write_file(
    path,
    bytes,
    create_mode=CreateNew,
    permission=0o600,
    sync=Full,
  ) catch {
    _ => raise ExportFailed(Write)
  }
}

///|
async fn cleanup_staging(staging : String) -> Bool noraise {
  let exists = @fs.exists(staging) catch { _ => return false }
  if !exists {
    return true
  }
  try @fs.rmdir(staging, recursive=true) catch {
    _ => false
  } noraise {
    _ => true
  }
}

///|
async fn write_candidate_envelope(
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
  staging : String,
) -> Bytes raise LunaBf16CandidateExportError {
  @fs.mkdir(staging + "/candidate-root", permission=0o700) catch {
    _ => raise ExportFailed(Create)
  }
  @fs.mkdir(staging + "/candidate-root/sources", permission=0o700) catch {
    _ => raise ExportFailed(Create)
  }
  @fs.mkdir(staging + "/candidate-root/recipes", permission=0o700) catch {
    _ => raise ExportFailed(Create)
  }
  write_new_file(
    staging + "/candidate-root/candidate-set.v1",
    set.candidate_set_bytes(),
  )
  for index in 0..<set.operation_count() {
    guard set.candidate_key(index) is Some(key) &&
      set.candidate_source_bytes(index) is Some(source) &&
      set.candidate_recipe_bytes(index) is Some(recipe) else {
      raise ExportFailed(CandidateSet)
    }
    write_new_file(staging + "/candidate-root/sources/" + key + ".cu", source)
    write_new_file(
      staging + "/candidate-root/recipes/" + key + ".recipe",
      recipe,
    )
  }
  write_new_file(staging + "/candidate.files.sha256", set.inventory_bytes())
  let record = export_record(set)
  write_new_file(staging + "/export.v1", record)
  record
}

///|
async fn export_new_protected(
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
  absolute_new_output : String,
) -> LunaBf16CandidateExportEvidence raise LunaBf16CandidateExportError {
  let staging = admit_new_output_path(absolute_new_output)
  // Cleanup is armed only after this invocation created the staging directory.
  // A failed/raced create therefore never removes a path owned by another actor.
  @fs.mkdir(staging, permission=0o700) catch {
    _ => raise ExportFailed(Create)
  }
  let record = try write_candidate_envelope(set, staging) catch {
    error => {
      if cleanup_staging(staging) {
        raise error
      }
      raise ExportFailedAndCleanup(Write, Cleanup)
    }
  } noraise {
    value => value
  }
  @fs.rename(staging, absolute_new_output, replace=false) catch {
    _ => {
      if cleanup_staging(staging) {
        raise ExportFailed(Publish)
      }
      raise ExportFailedAndCleanup(Publish, Cleanup)
    }
  }
  let identity = set.model_identity()
  {
    model_content_sha256: identity.content().as_hex(),
    model_plan_sha256: identity.plan().as_hex(),
    target: export_target_name(set),
    operation_count: set.operation_count(),
    candidate_set_sha256: set.candidate_set_sha256(),
    inventory_sha256: set.inventory_sha256(),
    export_record_sha256: export_sha256(record),
  }
}

///|
/// Atomically publishes one new offline compiler-input envelope. The output
/// parent must already be a canonical absolute directory. Existing outputs or
/// staging paths are never replaced. No compiler, process, device, runtime, or
/// input filesystem authority is accepted by this package.
pub async fn export_new(
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
  absolute_new_output : String,
) -> LunaBf16CandidateExportEvidence {
  @async.protect_from_cancel(() => {
    export_new_protected(set, absolute_new_output)
  })
}
