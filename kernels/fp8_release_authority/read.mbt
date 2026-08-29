///|
fn read_release_file(
  root : @approved_fs.ApprovedRoot,
  locator : @approved_fs.ApprovedRelativeLocator,
  maximum : Int64,
  rule : Fp8ReleaseAuthorityRule,
) -> Bytes raise Fp8ReleaseAuthorityError {
  let file = root.open_file(locator) catch {
    _ => raise InvalidFp8ReleaseAuthority(rule)
  }
  let bytes = file.read_immutable_snapshot(maximum_bytes=maximum) catch {
    _ => {
      file.close() catch {
        _ => ()
      }
      raise InvalidFp8ReleaseAuthority(rule)
    }
  }
  file.close() catch {
    _ => raise InvalidFp8ReleaseAuthority(rule)
  }
  bytes
}
