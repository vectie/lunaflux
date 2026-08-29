///|
async test "approved root requires exact absolute directory identity" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-approved-fs-identity")
    let parent = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(parent, recursive=true))
    }
    let admitted_path = "\{parent}/admitted"
    let other_path = "\{parent}/other"
    @fs.mkdir(admitted_path)
    @fs.mkdir(other_path)
    let root = @approved_fs.ApprovedRoot::open_absolute(admitted_path)
    root.require_absolute_identity(admitted_path)
    root.require_absolute_identity(admitted_path)
    try root.require_absolute_identity(other_path) catch {
      Invalid(RootIdentity) => ()
      _ => fail("unexpected mismatched root identity error")
    } noraise {
      _ => fail("different directory satisfied root identity")
    }
    root.close()
    try root.require_absolute_identity(admitted_path) catch {
      Invalid(Lifecycle) => ()
      _ => fail("unexpected closed root identity error")
    } noraise {
      _ => fail("closed root retained identity authority")
    }
  }
}

///|
async test "approved root identity follows no final or intermediate symlink" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-approved-fs-identity-link")
    let parent = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(parent, recursive=true))
    }
    let admitted_path = "\{parent}/admitted"
    let nested_path = "\{admitted_path}/nested"
    let link_path = "\{parent}/admitted.link"
    @fs.mkdir(nested_path, recursive=true)
    @fs.symlink(target=admitted_path, link_path)
    let root = @approved_fs.ApprovedRoot::open_absolute(admitted_path)
    for label in [link_path, "\{link_path}/nested"] {
      try root.require_absolute_identity(label) catch {
        UnsupportedType(Root) => ()
        _ => fail("unexpected symlink identity error")
      } noraise {
        _ => fail("symlink label satisfied root identity")
      }
    }
    root.close()
  }
}

///|
async test "approved root identity rejects namespace replacement but accepts moved label" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-approved-fs-identity-move")
    let parent = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(parent, recursive=true))
    }
    let admitted_path = "\{parent}/admitted"
    let moved_path = "\{parent}/moved"
    @fs.mkdir(admitted_path)
    let root = @approved_fs.ApprovedRoot::open_absolute(admitted_path)
    @fs.rename(admitted_path, moved_path)
    @fs.mkdir(admitted_path)
    try root.require_absolute_identity(admitted_path) catch {
      Invalid(RootIdentity) => ()
      _ => fail("unexpected replacement identity error")
    } noraise {
      _ => fail("hostile replacement satisfied pinned root identity")
    }
    root.require_absolute_identity(moved_path)
    assert_false(root.is_closed())
    root.close()
  }
}

///|
async test "root identity validates its absolute label before native traversal" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-approved-fs-label")
    let root_path = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root_path, recursive=true))
    }
    let root = @approved_fs.ApprovedRoot::open_absolute(root_path)
    for label in ["", "relative", "/tmp//root", "/tmp/../root"] {
      try root.require_absolute_identity(label) catch {
        Invalid(RootPath) => ()
        _ => fail("unexpected invalid identity label error")
      } noraise {
        _ => fail("invalid identity label reached traversal")
      }
    }
    root.close()
  }
}
