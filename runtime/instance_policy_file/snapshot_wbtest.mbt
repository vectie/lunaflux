///|
async test "policy loading retains the supplied label without a checksum scan" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-policy-label")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    @fs.write_file(
      "\{directory}/policy.json",
      @utf8.encode(policy_fixture()),
      create_mode=CreateNew,
    )
    let label = @fs.realpath(directory)
    let root = @approved_fs.ApprovedRoot::open_absolute(label)
    defer root.close()
    let digest = policy_test_digest()
    let limits = InstancePolicyFileLimits::new(
      max_policy_bytes=65536L,
      max_json_depth=16,
    )
    let policy = load(
      root,
      label,
      @approved_fs.ApprovedRelativeLocator::new("policy.json"),
      digest,
      limits,
    )
    assert_eq(policy.digest(), digest)
    assert_eq(policy.scheduler().max_active_requests(), 4)
    assert_false(root.is_closed())
    @fs.write_file(
      "\{directory}/malformed.json",
      b"not-json",
      create_mode=CreateNew,
    )
    try
      load(
        root,
        label,
        @approved_fs.ApprovedRelativeLocator::new("malformed.json"),
        digest,
        limits,
      )
    catch {
      MalformedJson => ()
      _ => fail("unexpected policy parse failure")
    } noraise {
      _ => fail("malformed policy was admitted")
    }
  }
}
