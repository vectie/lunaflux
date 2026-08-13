///|
test "approved filesystem paths are strict canonical ASCII" {
  for
    path in [
      "", "/", "relative", "/tmp//root", "/tmp/root/", "/tmp/../root", "/tmp\\root",
    ] {
    try @approved_fs.ApprovedRoot::open_absolute(path) catch {
      Invalid(RootPath) => ()
      _ => fail("unexpected invalid root error")
    } noraise {
      root => {
        root.close()
        fail("invalid root was opened")
      }
    }
  }
}

///|
async test "approved root and file pin exact objects across namespace replacement" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-approved-fs-pin")
    let parent = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(parent, recursive=true))
    }
    let root_path = "\{parent}/root"
    let moved_path = "\{parent}/moved"
    @fs.mkdir(root_path)
    @fs.write_file(
      "\{root_path}/model.bin",
      b"old",
      create_mode=CreateOrTruncate,
    )
    let root = @approved_fs.ApprovedRoot::open_absolute(root_path)
    @fs.rename(root_path, moved_path)
    @fs.mkdir(root_path)
    @fs.write_file(
      "\{root_path}/model.bin",
      b"new",
      create_mode=CreateOrTruncate,
    )
    let file = root.open_file("model.bin")
    let bytes = FixedArray::make(3, b'\x00')
    file.read_exactly_at(bytes, destination_offset=0, byte_count=3, position=0L)
    assert_eq(bytes, FixedArray::from_array([b'o', b'l', b'd']))
    file.close()
    root.close()
  }
}

///|
async test "approved file remains pinned after replacement and removal" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-approved-fs-file")
    let root_path = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root_path, recursive=true))
    }
    let path = "\{root_path}/weights.bin"
    let old_path = "\{root_path}/weights.old"
    @fs.write_file(path, b"first", create_mode=CreateOrTruncate)
    let root = @approved_fs.ApprovedRoot::open_absolute(root_path)
    let file = root.open_file("weights.bin")
    let stamp = file.stamp()
    assert_eq(stamp.byte_count(), 5L)
    @fs.rename(path, old_path)
    @fs.write_file(path, b"later", create_mode=CreateOrTruncate)
    @fs.remove(old_path)
    root.close()
    let bytes = FixedArray::make(7, b'!')
    file.read_exactly_at(bytes, destination_offset=1, byte_count=5, position=0L)
    assert_eq(bytes[0], b'!')
    assert_eq(bytes[1], b'f')
    assert_eq(bytes[2], b'i')
    assert_eq(bytes[3], b'r')
    assert_eq(bytes[4], b's')
    assert_eq(bytes[5], b't')
    assert_eq(bytes[6], b'!')
    file.close()
  }
}

///|
async test "approved traversal rejects symlinks and wrong file types" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-approved-fs-types")
    let parent = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(parent, recursive=true))
    }
    let root_path = "\{parent}/root"
    @fs.mkdir("\{root_path}/actual", recursive=true)
    @fs.write_file(
      "\{root_path}/actual/file.bin",
      b"x",
      create_mode=CreateOrTruncate,
    )
    @fs.symlink(target="\{root_path}/actual", "\{root_path}/linked")
    @fs.symlink(
      target="\{root_path}/actual/file.bin",
      "\{root_path}/final.link",
    )
    @fs.symlink(target=root_path, "\{parent}/root.link")
    let root = @approved_fs.ApprovedRoot::open_absolute(root_path)
    for locator in ["linked/file.bin", "final.link", "actual"] {
      try root.open_file(locator) catch {
        UnsupportedType(File) => ()
        _ => fail("unexpected unsafe file type error")
      } noraise {
        opened => {
          opened.close()
          fail("unsafe file type was opened")
        }
      }
    }
    try @approved_fs.ApprovedRoot::open_absolute("\{parent}/root.link") catch {
      UnsupportedType(Root) => ()
      _ => fail("unexpected final root symlink error")
    } noraise {
      opened => {
        opened.close()
        fail("root symlink was opened")
      }
    }
    let intermediate = "\{parent}/root.link/actual"
    try @approved_fs.ApprovedRoot::open_absolute(intermediate) catch {
      UnsupportedType(Root) => ()
      _ => fail("unexpected intermediate root symlink error")
    } noraise {
      opened => {
        opened.close()
        fail("intermediate root symlink was opened")
      }
    }
    root.close()
  }
}

///|
async test "relative locators and read ranges fail before native mutation" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-approved-fs-range")
    let root_path = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root_path, recursive=true))
    }
    @fs.write_file(
      "\{root_path}/file.bin",
      b"abc",
      create_mode=CreateOrTruncate,
    )
    let root = @approved_fs.ApprovedRoot::open_absolute(root_path)
    for
      locator in [
        "", "/file.bin", "../file.bin", "a//b", "a/./b", "file.bin/", "a\\b",
      ] {
      try root.open_file(locator) catch {
        Invalid(RelativeLocator) => ()
        _ => fail("unexpected invalid locator error")
      } noraise {
        opened => {
          opened.close()
          fail("invalid relative locator was opened")
        }
      }
    }
    let file = root.open_file("file.bin")
    assert_eq(file.read_immutable_snapshot(maximum_bytes=3L), b"abc")
    try file.read_immutable_snapshot(maximum_bytes=2L) catch {
      FileTooLarge => ()
      _ => fail("unexpected snapshot limit error")
    } noraise {
      _ => fail("oversized snapshot succeeded")
    }
    try file.read_immutable_snapshot(maximum_bytes=-1L) catch {
      Invalid(ReadRange) => ()
      _ => fail("unexpected negative snapshot limit error")
    } noraise {
      _ => fail("negative snapshot limit was accepted")
    }
    try
      file.read_exactly_at(
        FixedArray::make(2, b'\x00'),
        destination_offset=-1,
        byte_count=1,
        position=0L,
      )
    catch {
      Invalid(ReadRange) => ()
      _ => fail("unexpected read range error")
    } noraise {
      _ => fail("invalid read range was accepted")
    }
    try
      file.read_exactly_at(
        FixedArray::make(4, b'\x00'),
        destination_offset=0,
        byte_count=4,
        position=0L,
      )
    catch {
      Truncated => ()
      _ => fail("unexpected truncated read error")
    } noraise {
      _ => fail("truncated file read succeeded")
    }
    file.close()
    file.close()
    assert_true(file.is_closed())
    try file.stamp() catch {
      Invalid(Lifecycle) => ()
      _ => fail("unexpected closed file error")
    } noraise {
      _ => fail("closed file remained readable")
    }
    try file.read_immutable_snapshot(maximum_bytes=3L) catch {
      Invalid(Lifecycle) => ()
      _ => fail("unexpected closed snapshot error")
    } noraise {
      _ => fail("closed file published a snapshot")
    }
    root.close()
    root.close()
    assert_true(root.is_closed())
  }
}

///|
async test "independent same-root owners and close soak remain balanced" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-approved-fs-soak")
    let root_path = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root_path, recursive=true))
    }
    @fs.write_file("\{root_path}/file.bin", b"z", create_mode=CreateOrTruncate)
    for _ in 0..<256 {
      let left = @approved_fs.ApprovedRoot::open_absolute(root_path)
      let right = @approved_fs.ApprovedRoot::open_absolute(root_path)
      let left_file = left.open_file("file.bin")
      let right_file = right.open_file("file.bin")
      let byte = FixedArray::make(1, b'\x00')
      left_file.read_exactly_at(
        byte,
        destination_offset=0,
        byte_count=1,
        position=0L,
      )
      assert_eq(byte[0], b'z')
      left_file.close()
      right_file.close()
      left.close()
      right.close()
    }
  }
}
