///|
fn map_fs_error(error : @approved_fs.ApprovedFsError) -> LaunchFileError {
  match error {
    Unavailable(_) => FileUnavailable
    UnsupportedType(_) => UnsupportedFileType
    FileTooLarge => FileTooLarge
    Truncated => FileTruncated
    ChangedDuringRead => FileChangedDuringRead
    Busy(_) => FileBusy
    Invalid(_) | OperationFailed(_) => FileOperationFailed
  }
}

///|
/// Validates the literal one-argument grammar before any authority is opened.
fn parse_argument(
  argument : StringView,
) -> (String, LaunchFileDigest) raise LaunchFileError {
  let marker = "#sha256="
  let mut split = -1
  let mut index = 0
  while index + marker.length() <= argument.length() {
    if argument[index:index + marker.length()].to_owned() == marker {
      if split >= 0 {
        raise InvalidArgument
      }
      split = index
    }
    index += 1
  }
  if split <= 0 || split + marker.length() + 64 != argument.length() {
    raise InvalidArgument
  }
  let root = argument[0:split].to_owned()
  let digest = LaunchFileDigest::from_sha256(
    argument[split + marker.length():argument.length()],
  )
  validate_absolute_label(root) catch {
    _ => raise InvalidArgument
  }
  (root, digest)
}

///|
/// Loads the fixed launch descendant and closes all launch authority first.
pub fn load_argument(
  argument : StringView,
  limits : LaunchFileLimits,
) -> LaunchFileAdmission raise LaunchFileError {
  let (root_label, expected) = parse_argument(argument)
  let root = @approved_fs.ApprovedRoot::open_absolute(root_label) catch {
    error => raise map_fs_error(error)
  }
  let locator = @approved_fs.ApprovedRelativeLocator::new(
    "lunaflux.launch.json",
  ) catch {
    _ => {
      root.close() catch {
        _ => raise FileOperationFailed
      }
      raise FileOperationFailed
    }
  }
  let file = root.open_file(locator) catch {
    error => {
      root.close() catch {
        _ => raise FileOperationFailed
      }
      raise map_fs_error(error)
    }
  }
  let read : Result[Bytes, LaunchFileError] = Ok(
    file.read_immutable_snapshot(maximum_bytes=limits.maximum_bytes()),
  ) catch {
    error => Err(map_fs_error(error))
  }
  let file_close : Result[Unit, LaunchFileError] = Ok(file.close()) catch {
    error => Err(map_fs_error(error))
  }
  let root_close : Result[Unit, LaunchFileError] = Ok(root.close()) catch {
    error => Err(map_fs_error(error))
  }
  match read {
    Err(primary) => raise primary
    Ok(_) if file_close is Err(_) => raise FileOperationFailed
    Ok(_) if root_close is Err(_) => raise FileOperationFailed
    Ok(bytes) => {
      let actual = @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
      if actual != expected.as_hex() {
        raise DigestMismatch
      }
      parse_launch(bytes, expected, limits)
    }
  }
}
