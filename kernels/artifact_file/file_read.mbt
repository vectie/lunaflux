///|
priv struct FileStamp {
  byte_count : Int64
  modified_seconds : Int64
  modified_nanoseconds : Int
} derive(Eq)

///|
priv struct OpenedModuleFile {
  index : Int
  digest : @catalog.AotArtifactDigest
  file : @fs.File
  stamp : FileStamp
}

///|
async fn file_stamp(file : @fs.File) -> FileStamp {
  let byte_count = file.size()
  let (modified_seconds, modified_nanoseconds) = file.mtime()
  { byte_count, modified_seconds, modified_nanoseconds }
}

///|
async fn open_regular_file(
  root : StringView,
  identifier : StringView,
  file_ref : ArtifactFileRef,
) -> @fs.File raise KernelArtifactFileError {
  validate_path_components(root, identifier, file_ref)
  let file = @fs.open(join_path(root, identifier), mode=ReadOnly) catch {
    _ => raise FileUnavailable(file_ref)
  }
  if file.kind() != Regular {
    file.close()
    raise UnsupportedPathType(file_ref)
  }
  file
}

///|
async fn read_exact_snapshot(
  file : @fs.File,
  stamp : FileStamp,
  file_ref : ArtifactFileRef,
) -> Bytes raise KernelArtifactFileError {
  let bytes = file.read_exactly_at(stamp.byte_count.to_int(), position=0L) catch {
    _ => raise FileTruncated(file_ref)
  }
  if bytes.length().to_int64() != stamp.byte_count {
    raise FileTruncated(file_ref)
  }
  let trailing_probe = FixedArray::make(1, b'\x00')
  let trailing = file.read_at(
    trailing_probe,
    position=stamp.byte_count,
    offset=0,
    len=1,
  ) catch {
    _ => raise FileChangedDuringRead(file_ref)
  }
  if trailing != 0 {
    raise FileChangedDuringRead(file_ref)
  }
  let observed = file_stamp(file) catch {
    _ => raise FileChangedDuringRead(file_ref)
  }
  if observed != stamp {
    raise FileChangedDuringRead(file_ref)
  }
  bytes
}

///|
fn sha256_hex(bytes : BytesView) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn close_modules(files : ArrayView[OpenedModuleFile]) -> Unit {
  for opened in files {
    opened.file.close()
  }
}
