///|
fn validate_admitted_source(
  manifest : KernelArtifactManifest,
  limits : KernelArtifactFileLimits,
) -> Unit raise KernelArtifactFileError {
  let artifact_limits = limits.artifact()
  if manifest.module_digests.length() > artifact_limits.max_modules() {
    raise SourceCountExceeded(
      source=SourceModules,
      maximum=artifact_limits.max_modules(),
    )
  }
  if manifest.entry_points.length() > artifact_limits.max_entry_points() {
    raise SourceCountExceeded(
      source=SourceEntryPoints,
      maximum=artifact_limits.max_entry_points(),
    )
  }
  let seen_paths : Map[String, Bool] = Map([])
  for index, path in manifest.module_path_identifiers {
    validate_path_identifier(path, ModuleFile(index), limits.max_path_bytes())
    if seen_paths.contains(path) {
      raise InvalidPath(file=ModuleFile(index), issue=DuplicatePath)
    }
    seen_paths[path] = true
  }
}

///|
fn read_manifest_snapshot(
  root : @approved_fs.ApprovedRoot,
  identifier : StringView,
  expected : KernelManifestDigest,
  limits : KernelArtifactFileLimits,
) -> Bytes raise KernelArtifactFileError {
  let file = open_artifact_file(root, identifier, ManifestFile)
  let bytes = file.read_immutable_snapshot(
    maximum_bytes=limits.max_manifest_bytes(),
  ) catch {
    error => {
      let primary = snapshot_error(
        error,
        ManifestFile,
        limits.max_manifest_bytes(),
        false,
      )
      raise_primary_after_cleanup(primary, () => {
        close_artifact_file(file, ManifestFile)
      })
    }
  }
  close_artifact_file(file, ManifestFile)
  if sha256_hex(bytes) != expected.as_hex() {
    raise DigestMismatch(ManifestFile)
  }
  bytes
}

///|
fn open_module_files(
  root : @approved_fs.ApprovedRoot,
  manifest : KernelArtifactManifest,
  limits : KernelArtifactFileLimits,
  opened : Array[OpenedModuleFile],
) -> Unit raise KernelArtifactFileError {
  let artifact_limits = limits.artifact()
  let mut total = 0L
  for index in 0..<manifest.module_digests.length() {
    let file_ref = ModuleFile(index)
    let file = open_artifact_file(
      root,
      manifest.module_path_identifiers[index],
      file_ref,
    )
    let stamp = file.stamp() catch {
      error => {
        let primary = filesystem_error(error, file_ref, StatFile)
        raise_primary_after_cleanup(primary, () => {
          close_artifact_file(file, file_ref)
        })
      }
    }
    let byte_count = stamp.byte_count()
    if byte_count < 0L || byte_count > artifact_limits.max_module_bytes() {
      let primary = FileTooLarge(
        file=file_ref,
        maximum=artifact_limits.max_module_bytes(),
      )
      raise_primary_after_cleanup(primary, () => {
        close_artifact_file(file, file_ref)
      })
    }
    if byte_count > artifact_limits.max_total_module_bytes() - total {
      let primary = TotalModuleBytesExceeded(
        maximum=artifact_limits.max_total_module_bytes(),
      )
      raise_primary_after_cleanup(primary, () => {
        close_artifact_file(file, file_ref)
      })
    }
    total += byte_count
    opened.push({
      index,
      digest: manifest.module_digests[index],
      file,
      byte_count,
    })
  }
}

///|
fn read_module_inputs(
  opened : ArrayView[OpenedModuleFile],
) -> Array[@artifact.KernelModuleInput] raise KernelArtifactFileError {
  let inputs : Array[@artifact.KernelModuleInput] = []
  for item in opened {
    let file_ref = ModuleFile(item.index)
    let bytes = item.file.read_immutable_snapshot(maximum_bytes=item.byte_count) catch {
      error => raise snapshot_error(error, file_ref, item.byte_count, true)
    }
    if bytes.length().to_int64() != item.byte_count {
      raise FileChangedDuringRead(file_ref)
    }
    if sha256_hex(bytes) != item.digest.sha256() {
      raise DigestMismatch(file_ref)
    }
    inputs.push(
      @artifact.KernelModuleInput::from_bytes(
        digest=item.digest,
        module_bytes=bytes,
      ),
    )
  }
  inputs
}

///|
fn load_admitted_inputs(
  approved_root : @approved_fs.ApprovedRoot,
  manifest : KernelArtifactManifest,
  limits : KernelArtifactFileLimits,
) -> LoadedArtifactInputs raise KernelArtifactFileError {
  validate_admitted_source(manifest, limits)
  let opened : Array[OpenedModuleFile] = []
  let module_inputs = try {
    open_module_files(approved_root, manifest, limits, opened)
    read_module_inputs(opened)
  } catch {
    primary =>
      raise_primary_after_cleanup(primary, () => close_module_files(opened))
  } noraise {
    inputs => inputs
  }
  close_module_files(opened)
  { manifest, module_inputs }
}

///|
fn load_manifest_inputs(
  approved_root : @approved_fs.ApprovedRoot,
  manifest_identifier : StringView,
  expected_manifest_digest : KernelManifestDigest,
  evidence : ManifestContractEvidence,
  limits : KernelArtifactFileLimits,
) -> LoadedArtifactInputs raise KernelArtifactFileError {
  validate_path_identifier(
    manifest_identifier,
    ManifestFile,
    limits.max_path_bytes(),
  )
  let manifest_bytes = read_manifest_snapshot(
    approved_root, manifest_identifier, expected_manifest_digest, limits,
  )
  let manifest = read_manifest_with_evidence(manifest_bytes, evidence, limits)
  load_admitted_inputs(approved_root, manifest, limits)
}

///|
/// Loads and admits an exact digest-pinned AOT manifest through one caller-
/// owned pinned root. The root remains owned and open for the caller.
pub fn load(
  approved_root : @approved_fs.ApprovedRoot,
  manifest_identifier : StringView,
  expected_manifest_digest : KernelManifestDigest,
  contracts : @launch_contract.LaunchContractSet,
  limits : KernelArtifactFileLimits,
) -> @artifact.KernelArtifactBundle raise KernelArtifactFileError {
  let loaded = load_manifest_inputs(
    approved_root,
    manifest_identifier,
    expected_manifest_digest,
    stateless_contract_evidence(contracts),
    limits,
  )
  @artifact.admit(
    contracts,
    loaded.module_inputs,
    loaded.manifest.entry_points,
    limits.artifact(),
  ) catch {
    error => raise ArtifactRejected(error)
  }
}

///|
/// Loads an already admitted inert file source through one caller-owned root,
/// then performs stateless semantic artifact admission.
pub fn load_admitted(
  approved_root : @approved_fs.ApprovedRoot,
  source : KernelArtifactManifest,
  contracts : @launch_contract.LaunchContractSet,
  limits : KernelArtifactFileLimits,
) -> @artifact.KernelArtifactBundle raise KernelArtifactFileError {
  let loaded = load_admitted_inputs(approved_root, source, limits)
  @artifact.admit(
    contracts,
    loaded.module_inputs,
    loaded.manifest.entry_points,
    limits.artifact(),
  ) catch {
    error => raise ArtifactRejected(error)
  }
}

///|
/// Loads the exact digest-pinned filesystem format and admits its complete
/// paged launch-contract set without discovery, fallback, or compilation.
pub fn load_paged_kv(
  approved_root : @approved_fs.ApprovedRoot,
  manifest_identifier : StringView,
  expected_manifest_digest : KernelManifestDigest,
  contracts : @launch_contract.PagedKvLaunchContractSet,
  limits : KernelArtifactFileLimits,
) -> @artifact.KernelArtifactBundle raise KernelArtifactFileError {
  let loaded = load_manifest_inputs(
    approved_root,
    manifest_identifier,
    expected_manifest_digest,
    paged_contract_evidence(contracts),
    limits,
  )
  @artifact.admit_paged_kv(
    contracts,
    loaded.module_inputs,
    loaded.manifest.entry_points,
    limits.artifact(),
  ) catch {
    error => raise ArtifactRejected(error)
  }
}

///|
/// Loads an already admitted inert file source through one caller-owned root,
/// then performs paged semantic artifact admission.
pub fn load_paged_kv_admitted(
  approved_root : @approved_fs.ApprovedRoot,
  source : KernelArtifactManifest,
  contracts : @launch_contract.PagedKvLaunchContractSet,
  limits : KernelArtifactFileLimits,
) -> @artifact.KernelArtifactBundle raise KernelArtifactFileError {
  let loaded = load_admitted_inputs(approved_root, source, limits)
  @artifact.admit_paged_kv(
    contracts,
    loaded.module_inputs,
    loaded.manifest.entry_points,
    limits.artifact(),
  ) catch {
    error => raise ArtifactRejected(error)
  }
}

///|
fn sha256_hex(bytes : BytesView) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}
