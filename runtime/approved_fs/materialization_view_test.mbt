///|
async test "materialization view binds pinned source to inert target label" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-materialization-view")
    let parent = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(parent, recursive=true))
    }
    let source = "\{parent}/source"
    let moved = "\{parent}/moved"
    @fs.mkdir(source)
    @fs.write_file("\{source}/value.bin", b"old", create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(source)
    let view = root.materialization_view(source, "/opt/lunaflux/model")
    assert_eq(view.target_label(), "/opt/lunaflux/model")
    @fs.rename(source, moved)
    @fs.mkdir(source)
    @fs.write_file("\{source}/value.bin", b"new", create_mode=CreateNew)
    let locator = @approved_fs.ApprovedRelativeLocator::new("value.bin")
    let file = view.source_root().open_file(locator)
    assert_eq(file.read_immutable_snapshot(maximum_bytes=3L), b"old")
    file.close()
    root.close()
  }
}

///|
async test "materialization view rejects substituted source and hostile target" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-materialization-hostile")
    let parent = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(parent, recursive=true))
    }
    let source = "\{parent}/source"
    let other = "\{parent}/other"
    @fs.mkdir(source)
    @fs.mkdir(other)
    let root = @approved_fs.ApprovedRoot::open_absolute(source)
    try root.materialization_view(other, "/opt/lunaflux/model") catch {
      Invalid(RootIdentity) => ()
      _ => abort("substituted source failed at the wrong rule")
    } noraise {
      _ => abort("substituted source gained a materialization view")
    }
    for target in ["relative", "/opt/../model", "/opt/model/"] {
      try root.materialization_view(source, target) catch {
        Invalid(RootPath) => ()
        _ => abort("hostile target failed at the wrong rule")
      } noraise {
        _ => abort("hostile target gained a materialization view")
      }
    }
    root.close()
  }
}
