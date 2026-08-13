///|
priv struct OpenedModuleFile {
  index : Int
  digest : @catalog.AotArtifactDigest
  file : @approved_fs.ApprovedFile
  byte_count : Int64
}

///|
fn filesystem_error(
  error : @approved_fs.ApprovedFsError,
  file : ArtifactFileRef,
  operation : ArtifactFileOperation,
) -> KernelArtifactFileError {
  match error {
    Unavailable(_) => FileUnavailable(file)
    UnsupportedType(_) => UnsupportedPathType(file)
    Invalid(_) => FileIo(file~, operation~, issue=AuthorityClosed)
    Busy(_) => FileIo(file~, operation~, issue=Busy)
    Truncated => FileTruncated(file)
    ChangedDuringRead => FileChangedDuringRead(file)
    FileTooLarge => FileIo(file~, operation~, issue=Failed)
    OperationFailed(_) => FileIo(file~, operation~, issue=Failed)
  }
}

///|
fn open_artifact_file(
  root : @approved_fs.ApprovedRoot,
  identifier : StringView,
  file_ref : ArtifactFileRef,
) -> @approved_fs.ApprovedFile raise KernelArtifactFileError {
  let locator = @approved_fs.ApprovedRelativeLocator::new(identifier) catch {
    error => raise filesystem_error(error, file_ref, OpenFile)
  }
  root.open_file(locator) catch {
    error => raise filesystem_error(error, file_ref, OpenFile)
  }
}

///|
fn close_artifact_file(
  file : @approved_fs.ApprovedFile,
  file_ref : ArtifactFileRef,
) -> Unit raise KernelArtifactFileError {
  file.close() catch {
    error => raise filesystem_error(error, file_ref, CloseFile)
  }
}

///|
/// Runs deterministic cleanup after a primary failure without permitting a
/// secondary cleanup failure to replace the operation that actually failed.
fn[T] raise_primary_after_cleanup(
  primary : KernelArtifactFileError,
  cleanup : () -> Unit raise KernelArtifactFileError,
) -> T raise KernelArtifactFileError {
  try cleanup() catch {
    _ => raise primary
  } noraise {
    _ => raise primary
  }
}

///|
fn close_module_files(
  files : ArrayView[OpenedModuleFile],
) -> Unit raise KernelArtifactFileError {
  let first_failure : Ref[KernelArtifactFileError?] = Ref(None)
  for opened in files {
    opened.file.close() catch {
      error =>
        if first_failure.val is None {
          first_failure.val = Some(
            filesystem_error(error, ModuleFile(opened.index), CloseFile),
          )
        }
    }
  }
  match first_failure.val {
    Some(error) => raise error
    None => ()
  }
}

///|
fn snapshot_error(
  error : @approved_fs.ApprovedFsError,
  file : ArtifactFileRef,
  maximum : Int64,
  preflighted : Bool,
) -> KernelArtifactFileError {
  match error {
    FileTooLarge =>
      if preflighted {
        FileChangedDuringRead(file)
      } else {
        FileTooLarge(file~, maximum~)
      }
    _ => filesystem_error(error, file, SnapshotFile)
  }
}
