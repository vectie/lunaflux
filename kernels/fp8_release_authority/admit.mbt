///|
fn release_sha256(bytes : BytesView) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
pub fn admit_v2(
  kernel_root : @approved_fs.ApprovedRoot,
  manifest_locator : @approved_fs.ApprovedRelativeLocator,
  approval : @luna_capability_manifest.LunaAdmittedFp8ReleaseManifestV2,
  recipe : @fp8_runtime_recipe.Fp8RuntimeRecipeV2,
  limits : Fp8ReleaseAuthorityLimits,
) -> Fp8ReleaseAuthorityV2 raise Fp8ReleaseAuthorityError {
  let manifest = approval.manifest()
  let manifest_bytes = read_release_file(
    kernel_root,
    manifest_locator,
    limits.max_manifest_bytes,
    ManifestFile,
  )
  if manifest_bytes != manifest.signing_bytes() ||
    release_sha256(manifest_bytes) != manifest.digest().as_hex() {
    raise InvalidFp8ReleaseAuthority(ExternalApproval)
  }
  let startup = recipe.startup()
  if manifest.runtime_recipe_digest_sha256() != recipe.digest().as_hex() {
    raise InvalidFp8ReleaseAuthority(Recipe)
  }
  if manifest.model_identity() != startup.model_plan().identity() {
    raise InvalidFp8ReleaseAuthority(ModelIdentity)
  }
  let target = startup.kernel_artifacts().manifest().target()
  if manifest.target() != target {
    raise InvalidFp8ReleaseAuthority(Target)
  }
  let launches = recipe.launches()
  let claims = manifest.operations()
  if launches.is_empty() ||
    launches.length() != claims.length() ||
    launches.length() > limits.max_operations {
    raise InvalidFp8ReleaseAuthority(OperationOrder)
  }
  let artifacts = startup.kernel_artifacts().artifacts()
  let operations : Array[Fp8ReleaseOperationAuthorityV2] = []
  let module_digests : Array[String] = []
  let module_snapshots : Array[Bytes] = []
  let mut total_module_bytes = 0L
  for index, launch in launches {
    let claim = claims[index]
    if claim.operation() != launch.operation() {
      raise InvalidFp8ReleaseAuthority(OperationOrder)
    }
    if claim.entry_point() != launch.entry_point() {
      raise InvalidFp8ReleaseAuthority(EntryPoint)
    }
    guard artifacts.entry_point(launch.entry_point()) is Some(entry) &&
      artifacts.find_module(launch.module_digest()) is Some(artifact_module) else {
      raise InvalidFp8ReleaseAuthority(EntryPoint)
    }
    if claim.function_symbol() != entry.function_symbol() {
      raise InvalidFp8ReleaseAuthority(Symbol)
    }
    if claim.dimensions() != launch.launch_dimensions() {
      raise InvalidFp8ReleaseAuthority(LaunchDimensions)
    }
    if claim.operands() != launch.raw_operands() {
      raise InvalidFp8ReleaseAuthority(RawOperands)
    }
    if claim.workspace_bytes() != launch.workspace().byte_length() ||
      claim.workspace_alignment() != launch.workspace().alignment() {
      raise InvalidFp8ReleaseAuthority(Workspace)
    }
    let module_digest = launch.module_digest().sha256()
    let mut module_snapshot_index : Int? = None
    for snapshot_index, retained_digest in module_digests {
      if retained_digest == module_digest {
        module_snapshot_index = Some(snapshot_index)
        break
      }
    }
    let module_bytes = match module_snapshot_index {
      Some(snapshot_index) => module_snapshots[snapshot_index]
      None => {
        let locator = @approved_fs.ApprovedRelativeLocator::new(
          claim.module_locator(),
        ) catch {
          _ => raise InvalidFp8ReleaseAuthority(ModuleFile)
        }
        let snapshot = read_release_file(
          kernel_root,
          locator,
          limits.max_module_bytes,
          ModuleFile,
        )
        let length = snapshot.length().to_int64()
        if length > limits.max_total_module_bytes - total_module_bytes {
          raise InvalidFp8ReleaseAuthority(TotalModuleBytes)
        }
        total_module_bytes += length
        module_digests.push(module_digest)
        module_snapshots.push(snapshot)
        snapshot
      }
    }
    if release_sha256(module_bytes) != module_digest {
      raise InvalidFp8ReleaseAuthority(ModuleDigest)
    }
    if module_bytes != artifact_module.module_bytes() {
      raise InvalidFp8ReleaseAuthority(ArtifactBytes)
    }
    operations.push({
      operation: launch.operation(),
      entry_point: launch.entry_point(),
      function_symbol: entry.function_symbol(),
      module_bytes,
      source_digest_sha256: claim.source_digest_sha256(),
      recipe_digest_sha256: claim.recipe_digest_sha256(),
      dimensions: launch.launch_dimensions(),
      operands: launch.raw_operands(),
      workspace_bytes: launch.workspace().byte_length(),
      workspace_alignment: launch.workspace().alignment(),
    })
  }
  let retained = ReadOnlyArray::from_array(operations)
  {
    recipe,
    approval,
    operations: retained,
    // The externally approved manifest already canonically binds the exact
    // runtime-recipe digest and every operation/file identity.
    digest: Fp8ReleaseAuthorityDigest(manifest.digest().as_hex()),
  }
}
