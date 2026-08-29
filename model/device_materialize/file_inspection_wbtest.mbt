///|
fn wb_weight_locator(
  value : StringView,
) -> @approved_fs.ApprovedRelativeLocator raise @approved_fs.ApprovedFsError {
  @approved_fs.ApprovedRelativeLocator::new(value)
}

///|
fn wb_inspect_file(file : Bytes) -> DeviceWeightFileInspection raise {
  let model = wb_stream_model(file)
  let limits = wb_stream_limits(transfer_chunk_bytes=23)
  let prepared = prepare_streaming_load(
    model,
    @llama.build(model),
    wb_stamp(file),
    file.length().to_int64(),
    limits,
    64L,
    4096L,
    (destination, destination_offset, position, count) => {
      wb_fill(file, destination, destination_offset, position, count)
    },
    () => wb_stamp(file),
  )
  inspection_from_prepared(
    model,
    @llama.build(model),
    wb_weight_locator("weights.safetensors"),
    limits,
    64L,
    4096L,
    prepared,
  )
}

///|
async test "public inspection admits an exact file without a device context" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-weight-inspection")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let file = wb_stream_file()
    let path = "\{directory}/weights.safetensors"
    @fs.write_file(path, file, create_mode=CreateNew)
    let model = wb_stream_model(file)
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    let inspected = inspect_file(
      root,
      model,
      @llama.build(model),
      wb_weight_locator("weights.safetensors"),
      limits=wb_stream_limits(),
      alignment_bytes=64L,
      max_arena_bytes=4096L,
    )
    assert_eq(inspected.content_digest(), model.content_digest())
    assert_eq(inspected.layout().arena_bytes(), 768L)
    root.close()
  }
}

///|
async test "inspection file failures retain no locator payload" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-weight-missing")
    let root_path = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root_path, recursive=true))
    }
    let root = @approved_fs.ApprovedRoot::open_absolute(root_path)
    try
      inspect_file(
        root,
        wb_stream_model(wb_stream_file()),
        wb_stream_plan(wb_stream_file()),
        wb_weight_locator("private-model-name.safetensors"),
        limits=wb_stream_limits(),
        alignment_bytes=64L,
        max_arena_bytes=4096L,
      )
    catch {
      FileUnavailable => ()
      _ => fail("unexpected missing-file error")
    } noraise {
      _ => fail("expected missing-file rejection")
    }
    root.close()
  }
}

///|
async test "inspection rejects a non-regular locator" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-weight-nonregular")
    let root_path = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root_path, recursive=true))
    }
    @fs.mkdir("\{root_path}/directory")
    let root = @approved_fs.ApprovedRoot::open_absolute(root_path)
    try
      inspect_file(
        root,
        wb_stream_model(wb_stream_file()),
        wb_stream_plan(wb_stream_file()),
        wb_weight_locator("directory"),
        limits=wb_stream_limits(),
        alignment_bytes=64L,
        max_arena_bytes=4096L,
      )
    catch {
      UnsupportedFileType => ()
      _ => fail("unexpected non-regular error")
    } noraise {
      _ => fail("expected non-regular rejection")
    }
    root.close()
  }
}

///|
async test "inspection rejects symlink locators without payloads" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-weight-locator")
    let root_path = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root_path, recursive=true))
    }
    let file = wb_stream_file()
    @fs.write_file(
      "\{root_path}/weights.safetensors",
      file,
      create_mode=CreateNew,
    )
    @fs.symlink(
      target="\{root_path}/weights.safetensors",
      "\{root_path}/weights.link",
    )
    let root = @approved_fs.ApprovedRoot::open_absolute(root_path)
    try
      inspect_file(
        root,
        wb_stream_model(file),
        wb_stream_plan(file),
        wb_weight_locator("weights.link"),
        limits=wb_stream_limits(),
        alignment_bytes=64L,
        max_arena_bytes=4096L,
      )
    catch {
      UnsupportedFileType => ()
      _ => fail("unexpected symlink-locator error")
    } noraise {
      _ => fail("symlink locator was admitted")
    }
    root.close()
  }
}

///|
async test "inspection reads from the pinned root after namespace replacement" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-weight-pinned-root")
    let parent = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(parent, recursive=true))
    }
    let root_path = "\{parent}/root"
    let moved_path = "\{parent}/moved"
    let file = wb_stream_file()
    @fs.mkdir(root_path)
    @fs.write_file(
      "\{root_path}/weights.safetensors",
      file,
      create_mode=CreateNew,
    )
    let root = @approved_fs.ApprovedRoot::open_absolute(root_path)
    @fs.rename(root_path, moved_path)
    @fs.mkdir(root_path)
    @fs.write_file(
      "\{root_path}/weights.safetensors",
      b"replacement",
      create_mode=CreateNew,
    )
    let inspection = inspect_file(
      root,
      wb_stream_model(file),
      wb_stream_plan(file),
      wb_weight_locator("weights.safetensors"),
      limits=wb_stream_limits(),
      alignment_bytes=64L,
      max_arena_bytes=4096L,
    )
    assert_eq(
      inspection.content_digest(),
      wb_stream_model(file).content_digest(),
    )
    root.close()
  }
}

///|
test "weight inspection exposes exact immutable layout evidence" {
  let file = wb_stream_file()
  let inspection = wb_inspect_file(file)
  let model = wb_stream_model(file)
  let plan = @llama.build(model)
  assert_eq(inspection.content_digest(), model.content_digest())
  assert_eq(inspection.layout().identity(), plan.identity())
  assert_eq(inspection.layout().length(), 12)
  assert_eq(inspection.layout().arena_bytes(), 768L)
}

///|
test "readmission hashes exact bytes instead of trusting old file stamps" {
  let file = wb_stream_file()
  let inspection = wb_inspect_file(file)
  let current_stamp = wb_stamp(file, modified=17)
  let prepared = readmit_inspection(
    inspection,
    current_stamp,
    file.length().to_int64(),
    (destination, destination_offset, position, count) => {
      wb_fill(file, destination, destination_offset, position, count)
    },
    () => current_stamp,
  )
  assert_eq(prepared.layout, inspection.layout())
  assert_true(prepared.source_stamp == current_stamp)
}

///|
test "readmission rejects changed content before device allocation" {
  let file = wb_stream_file()
  let inspection = wb_inspect_file(file)
  let changed_data = file.to_array()
  let last = changed_data.length() - 1
  changed_data[last] = (changed_data[last].to_int() ^ 1).to_byte()
  let changed = Bytes::from_array(changed_data)
  let opens = Ref(0)
  try
    load_inspected_streaming_with_resource(
      inspection,
      wb_stamp(changed, modified=19),
      changed.length().to_int64(),
      (destination, destination_offset, position, count) => {
        wb_fill(changed, destination, destination_offset, position, count)
      },
      () => wb_stamp(changed, modified=19),
      _arena_bytes => {
        opens.val += 1
        1
      },
      (_resource, _source, _source_offset, _destination, _count) => (),
      _resource => (),
    )
  catch {
    ContentDigestMismatch => ()
    _ => fail("unexpected changed-content error")
  } noraise {
    _ => fail("expected changed-content rejection")
  }
  assert_eq(opens.val, 0)
}

///|
test "ctime-only same-handle changes reject inspection admission" {
  let file = wb_stream_file()
  try
    prepare_streaming_load(
      wb_stream_model(file),
      wb_stream_plan(file),
      wb_stamp(file, modified=7, changed=7),
      file.length().to_int64(),
      wb_stream_limits(),
      64L,
      4096L,
      (destination, destination_offset, position, count) => {
        wb_fill(file, destination, destination_offset, position, count)
      },
      () => wb_stamp(file, modified=7, changed=8),
    )
  catch {
    FileChangedDuringLoad => ()
    _ => fail("unexpected ctime-change error")
  } noraise {
    _ => fail("expected ctime-change rejection")
  }
}

///|
test "readmission requires exact semantic transfer evidence" {
  let file = wb_stream_file()
  let admitted = wb_inspect_file(file)
  let altered : DeviceWeightFileInspection = {
    model: admitted.model,
    model_plan: admitted.model_plan,
    locator: admitted.locator,
    limits: admitted.limits,
    alignment_bytes: admitted.alignment_bytes,
    max_arena_bytes: admitted.max_arena_bytes,
    payload_offset_bytes: admitted.payload_offset_bytes + 1L,
    layout: admitted.layout,
    regions: admitted.regions,
  }
  try
    readmit_inspection(
      altered,
      wb_stamp(file),
      file.length().to_int64(),
      (destination, destination_offset, position, count) => {
        wb_fill(file, destination, destination_offset, position, count)
      },
      () => wb_stamp(file),
    )
  catch {
    FileChangedDuringLoad => ()
    _ => fail("unexpected evidence-mismatch error")
  } noraise {
    _ => fail("expected evidence-mismatch rejection")
  }
}
