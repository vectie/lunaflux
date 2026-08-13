///|
async fn read_manifest_snapshot(
  root : StringView,
  identifier : StringView,
  expected : KernelManifestDigest,
  limits : KernelArtifactFileLimits,
) -> Bytes raise KernelArtifactFileError {
  let file = open_regular_file(root, identifier, ManifestFile)
  defer file.close()
  let stamp = file_stamp(file) catch {
    _ => raise FileUnavailable(ManifestFile)
  }
  if stamp.byte_count < 0L || stamp.byte_count > limits.max_manifest_bytes() {
    raise FileTooLarge(file=ManifestFile, maximum=limits.max_manifest_bytes())
  }
  let bytes = read_exact_snapshot(file, stamp, ManifestFile)
  if sha256_hex(bytes) != expected.as_hex() {
    raise DigestMismatch(ManifestFile)
  }
  bytes
}

///|
async fn open_module_files(
  root : StringView,
  manifest : KernelArtifactManifest,
  limits : KernelArtifactFileLimits,
  opened : Array[OpenedModuleFile],
) -> Unit raise KernelArtifactFileError {
  let artifact_limits = limits.artifact()
  let mut total = 0L
  for index in 0..<manifest.module_digests.length() {
    let file_ref = ModuleFile(index)
    let file = open_regular_file(
      root,
      manifest.module_path_identifiers[index],
      file_ref,
    )
    let stamp = file_stamp(file) catch {
      _ => {
        file.close()
        raise FileUnavailable(file_ref)
      }
    }
    if stamp.byte_count < 0L ||
      stamp.byte_count > artifact_limits.max_module_bytes() {
      file.close()
      raise FileTooLarge(
        file=file_ref,
        maximum=artifact_limits.max_module_bytes(),
      )
    }
    if stamp.byte_count > artifact_limits.max_total_module_bytes() - total {
      file.close()
      raise TotalModuleBytesExceeded(
        maximum=artifact_limits.max_total_module_bytes(),
      )
    }
    total += stamp.byte_count
    opened.push({ index, digest: manifest.module_digests[index], file, stamp })
  }
}

///|
async fn read_module_inputs(
  opened : ArrayView[OpenedModuleFile],
) -> Array[@artifact.KernelModuleInput] raise KernelArtifactFileError {
  let inputs : Array[@artifact.KernelModuleInput] = []
  for item in opened {
    let file_ref = ModuleFile(item.index)
    let bytes = read_exact_snapshot(item.file, item.stamp, file_ref)
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
async fn load_inputs(
  approved_root : StringView,
  manifest_identifier : StringView,
  expected_manifest_digest : KernelManifestDigest,
  evidence : ManifestContractEvidence,
  limits : KernelArtifactFileLimits,
) -> LoadedArtifactInputs raise KernelArtifactFileError {
  let root_kind = @fs.kind(approved_root, follow_symlink=false) catch {
    _ => raise ApprovedRootUnavailable
  }
  if root_kind != Directory {
    raise UnsupportedApprovedRootType
  }
  validate_path_identifier(
    manifest_identifier,
    ManifestFile,
    limits.max_path_bytes(),
  )
  let manifest_bytes = read_manifest_snapshot(
    approved_root, manifest_identifier, expected_manifest_digest, limits,
  )
  let manifest = read_manifest_with_evidence(manifest_bytes, evidence, limits)
  let opened : Array[OpenedModuleFile] = []
  defer close_modules(opened)
  open_module_files(approved_root, manifest, limits, opened)
  let module_inputs = read_module_inputs(opened)
  { manifest, module_inputs }
}

///|
/// Loads and admits an exact digest-pinned AOT manifest and its module files
/// from one caller-approved read-only artifact root.
///
/// The manifest identifier and all manifest module paths are strict relative
/// identifiers. Absolute paths, traversal, aliases, symlinks, directories,
/// pipes, sockets, and devices are rejected. All module handles are opened and
/// the aggregate size is proven before any module snapshot is allocated. Each
/// exact snapshot is then hashed through that same opened handle and delegated
/// to kernels/artifact admission for semantic validation.
///
/// The portable filesystem seam cannot make component traversal and opening
/// atomic. Deployments must therefore provide a stable approved read-only
/// mount; this function does not claim openat/O_NOFOLLOW race exclusion.
pub async fn load(
  approved_root : StringView,
  manifest_identifier : StringView,
  expected_manifest_digest : KernelManifestDigest,
  contracts : @launch_contract.LaunchContractSet,
  limits : KernelArtifactFileLimits,
) -> @artifact.KernelArtifactBundle raise KernelArtifactFileError {
  let loaded = load_inputs(
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
/// Loads the same exact digest-pinned filesystem format as `load`, then admits
/// it against a paged launch-contract set. No discovery, fallback, or runtime
/// compilation is performed.
pub async fn load_paged_kv(
  approved_root : StringView,
  manifest_identifier : StringView,
  expected_manifest_digest : KernelManifestDigest,
  contracts : @launch_contract.PagedKvLaunchContractSet,
  limits : KernelArtifactFileLimits,
) -> @artifact.KernelArtifactBundle raise KernelArtifactFileError {
  let loaded = load_inputs(
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
