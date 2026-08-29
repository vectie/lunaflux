///|
fn expect_release_read_rule(
  root : @approved_fs.ApprovedRoot,
  locator : String,
  maximum : Int64,
  expected : Fp8ReleaseAuthorityRule,
) -> Unit raise {
  let approved_locator = try! @approved_fs.ApprovedRelativeLocator::new(locator)
  try read_release_file(root, approved_locator, maximum, expected) catch {
    InvalidFp8ReleaseAuthority(rule) => assert_eq(rule, expected)
  } noraise {
    _ => fail("hostile release file was admitted")
  }
}

///|
async test "release file inspection is bounded and rejects symlink traversal" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-fp8-release-authority")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    @fs.write_file(
      "\{directory}/module.cubin",
      b"immutable-cubin",
      create_mode=CreateNew,
    )
    @fs.symlink(target="\{directory}/module.cubin", "\{directory}/module.link")
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    assert_eq(
      read_release_file(
        root,
        @approved_fs.ApprovedRelativeLocator::new("module.cubin"),
        64L,
        ModuleFile,
      ),
      b"immutable-cubin",
    )
    expect_release_read_rule(root, "module.link", 64L, ModuleFile)
    expect_release_read_rule(root, "module.cubin", 4L, ModuleFile)
    expect_release_read_rule(root, "missing.cubin", 64L, ModuleFile)
  }
}

///|
test "release authority limits reject overflow-prone bounds" {
  let valid = try! Fp8ReleaseAuthorityLimits::new(
    max_manifest_bytes=4096L,
    max_module_bytes=1048576L,
    max_total_module_bytes=2097152L,
    max_operations=8,
  )
  ignore(valid)
  for
    body in [
      () => {
        Fp8ReleaseAuthorityLimits::new(
          max_manifest_bytes=0L,
          max_module_bytes=1L,
          max_total_module_bytes=1L,
          max_operations=1,
        )
      },
      () => {
        Fp8ReleaseAuthorityLimits::new(
          max_manifest_bytes=1L,
          max_module_bytes=1024L * 1024L * 1024L + 1L,
          max_total_module_bytes=2147483647L,
          max_operations=1,
        )
      },
      () => {
        Fp8ReleaseAuthorityLimits::new(
          max_manifest_bytes=1L,
          max_module_bytes=1L,
          max_total_module_bytes=1L,
          max_operations=65537,
        )
      },
    ] {
    try body() catch {
      InvalidFp8ReleaseAuthority(Limit) => ()
      _ => fail("unexpected release authority limit rejection")
    } noraise {
      _ => fail("overflow-prone release authority limit was admitted")
    }
  }
}
