///|
pub(all) suberror ProjectionTuningFileError {
  InvalidTuningInput
  TuningReadFailure
} derive(Debug, Eq)

///|
pub struct ProjectionTuningInput {
  priv root : String
  priv locator : @approved_fs.ApprovedRelativeLocator
  priv sha256 : String
  priv device_identity : String
}

///|
/// A shared optional CLI suffix keeps exporter and binder input semantics equal.
pub fn parse_projection_tuning_suffix(
  suffix : ArrayView[String],
) -> ProjectionTuningInput? raise ProjectionTuningFileError {
  if suffix.is_empty() {
    return None
  }
  guard suffix
    is ["--projection-tuning", root, locator, sha256, device_identity] else {
    raise InvalidTuningInput
  }
  if !root.has_prefix("/") ||
    root == "/" ||
    sha256.length() != 64 ||
    !sha256.iter().all(c => c is ('0'..='9' | 'a'..='f')) ||
    device_identity.is_empty() ||
    device_identity.length() > 128 ||
    !device_identity
    .iter()
    .all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' | ':')) {
    raise InvalidTuningInput
  }
  Some({
    root,
    locator: @approved_fs.ApprovedRelativeLocator::new(locator),
    sha256,
    device_identity,
  }) catch {
    _ => raise InvalidTuningInput
  }
}

///|
/// Offline-only read: no root or file capability escapes to generated execution.
pub fn load_projection_tuning_records(
  input : ProjectionTuningInput?,
  backend~ : @strategy.ProjectionBackendId,
  architecture~ : @strategy.ProjectionTargetId,
  toolchain_sha256~ : String,
) -> ReadOnlyArray[@strategy.ProjectionAutotuneRecord] raise ProjectionTuningFileError {
  guard input is Some(input) else { return ReadOnlyArray::from_array([]) }
  try {
    let scope = @tuning.ProjectionTuningScope::new(
      backend~,
      architecture~,
      device_identity=input.device_identity,
      toolchain_sha256~,
    )
    let root = @approved_fs.ApprovedRoot::open_absolute(input.root)
    let read = try {
      let file = root.open_file(input.locator)
      let snapshot = Ok(file.read_immutable_snapshot(maximum_bytes=1048576L)) catch {
        error => Err(error)
      }
      file.close()
      match snapshot {
        Ok(bytes) => Ok(bytes)
        Err(_) => Err(TuningReadFailure)
      }
    } catch {
      _ => Err(TuningReadFailure)
    }
    root.close()
    let bytes = match read {
      Ok(bytes) => bytes
      Err(error) => raise error
    }
    if @crypto.bytes_to_hex_string(@crypto.sha256(bytes)) != input.sha256 {
      raise TuningReadFailure
    }
    @tuning.parse_projection_tuning_snapshot(bytes).records_for(scope)
  } catch {
    _ => raise TuningReadFailure
  }
}
