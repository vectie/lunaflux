///|
async fn fused_export_write_new(
  path : String,
  bytes : Bytes,
) -> Unit raise LunaFusedCandidateExportError {
  @fs.write_file(
    path,
    bytes,
    create_mode=CreateNew,
    permission=0o600,
    sync=Full,
  ) catch {
    _ => raise FusedCandidateExportFailed(Write)
  }
}

///|
async fn fused_export_cleanup(staging : String) -> Bool noraise {
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
fn fused_export_record(set : LunaFusedCandidateQualificationSet) -> Bytes {
  @utf8.encode(
    "schema=lunaflux-fused-candidate-export.v1\n" +
    "target=sm_120\nblock_threads=128\ncandidate_count=2\n" +
    "candidate_set_sha256=" +
    set.declaration_sha256 +
    "\ncandidate_inventory_sha256=" +
    set.inventory_sha256 +
    "\ncandidate_root=candidate-root\n" +
    "candidate_inventory=candidate.files.sha256\n" +
    "qualification_only=1\nmanifest_bindable=0\n" +
    "compiler_invoked=0\nartifact_authority=0\ndevice_opened=0\n" +
    "runtime_authority=0\npromotion_authority=0\n",
  )
}

///|
async fn fused_export_write_envelope(
  set : LunaFusedCandidateQualificationSet,
  staging : String,
) -> Bytes raise LunaFusedCandidateExportError {
  @fs.mkdir(staging + "/candidate-root", permission=0o700) catch {
    _ => raise FusedCandidateExportFailed(Create)
  }
  @fs.mkdir(staging + "/candidate-root/sources", permission=0o700) catch {
    _ => raise FusedCandidateExportFailed(Create)
  }
  @fs.mkdir(staging + "/candidate-root/recipes", permission=0o700) catch {
    _ => raise FusedCandidateExportFailed(Create)
  }
  fused_export_write_new(
    staging + "/candidate-root/candidate-set.v1",
    set.declaration,
  )
  fused_export_write_new(
    staging + "/candidate-root/sources/" + set.qkv_key + ".cu",
    set.qkv.source_bytes(),
  )
  fused_export_write_new(
    staging + "/candidate-root/recipes/" + set.qkv_key + ".recipe",
    set.qkv.recipe_bytes(),
  )
  fused_export_write_new(
    staging + "/candidate-root/sources/" + set.residual_norm_key + ".cu",
    set.residual_norm.source_bytes(),
  )
  fused_export_write_new(
    staging + "/candidate-root/recipes/" + set.residual_norm_key + ".recipe",
    set.residual_norm.recipe_bytes(),
  )
  fused_export_write_new(
    staging + "/candidate-root/sources/oracle_qkv_projection.cu",
    set.qkv_fallback.source_bytes(),
  )
  fused_export_write_new(
    staging + "/candidate-root/recipes/oracle_qkv_projection.recipe",
    set.qkv_fallback.recipe_bytes(),
  )
  fused_export_write_new(
    staging + "/candidate-root/sources/oracle_positioned_rope.cu",
    set.rope_fallback.source_bytes(),
  )
  fused_export_write_new(
    staging + "/candidate-root/recipes/oracle_positioned_rope.recipe",
    set.rope_fallback.recipe_bytes(),
  )
  fused_export_write_new(
    staging + "/candidate-root/sources/oracle_paged_kv_write.cu",
    set.kv_write_fallback.source_bytes(),
  )
  fused_export_write_new(
    staging + "/candidate-root/recipes/oracle_paged_kv_write.recipe",
    set.kv_write_fallback.recipe_bytes(),
  )
  fused_export_write_new(staging + "/candidate.files.sha256", set.inventory)
  let record = fused_export_record(set)
  fused_export_write_new(staging + "/export.v1", record)
  record
}

///|
async fn fused_export_new_protected(
  set : LunaFusedCandidateQualificationSet,
  absolute_new_output : String,
) -> LunaFusedCandidateExportEvidence raise LunaFusedCandidateExportError {
  let staging = fused_export_staging_path(absolute_new_output)
  @fs.mkdir(staging, permission=0o700) catch {
    _ => raise FusedCandidateExportFailed(Create)
  }
  let record = try fused_export_write_envelope(set, staging) catch {
    error => {
      if fused_export_cleanup(staging) {
        raise error
      }
      raise FusedCandidateExportFailedAndCleanup(Write, Cleanup)
    }
  } noraise {
    value => value
  }
  @fs.rename(staging, absolute_new_output, replace=false) catch {
    _ => {
      if fused_export_cleanup(staging) {
        raise FusedCandidateExportFailed(Publish)
      }
      raise FusedCandidateExportFailedAndCleanup(Publish, Cleanup)
    }
  }
  {
    declaration_sha256: set.declaration_sha256,
    inventory_sha256: set.inventory_sha256,
    export_record_sha256: fused_export_sha256(record),
  }
}

///|
/// Atomically publishes a source-owned, qualification-only sm120 candidate
/// envelope. Existing outputs are never replaced. This function compiles or
/// executes nothing and grants no artifact, manifest, runtime, or promotion
/// authority.
pub async fn export_new(
  set : LunaFusedCandidateQualificationSet,
  absolute_new_output : String,
) -> LunaFusedCandidateExportEvidence {
  @async.protect_from_cancel(() => {
    fused_export_new_protected(set, absolute_new_output)
  })
}
