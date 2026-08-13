///|
async fn wb_inspect_file(file : Bytes) -> DeviceWeightFileInspection {
  let model = wb_stream_model(file)
  let limits = wb_stream_limits(transfer_chunk_bytes=23)
  let prepared = prepare_streaming_load(
    model,
    wb_stamp(file),
    limits,
    64L,
    4096L,
    (position, count) => wb_slice(file, position, count),
    () => wb_stamp(file),
  )
  inspection_from_prepared(model, limits, 64L, 4096L, prepared)
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
    let inspected = inspect_file(
      model,
      path,
      limits=wb_stream_limits(),
      alignment_bytes=64L,
      max_arena_bytes=4096L,
    )
    assert_eq(inspected.content_digest(), model.identity().content())
    assert_eq(inspected.layout().arena_bytes(), 768L)
  }
}

///|
async test "inspection file failures retain no locator payload" {
  let locator = "/definitely/missing/private-model-name.safetensors"
  try
    inspect_file(
      wb_stream_model(wb_stream_file()),
      locator,
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
}

///|
async test "inspection rejects a non-regular locator" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-weight-nonregular")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    try
      inspect_file(
        wb_stream_model(wb_stream_file()),
        directory,
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
  }
}

///|
async test "weight inspection exposes exact immutable layout evidence" {
  let file = wb_stream_file()
  let inspection = wb_inspect_file(file)
  let model = wb_stream_model(file)
  assert_eq(inspection.content_digest(), model.identity().content())
  assert_eq(inspection.layout().identity(), model.identity())
  assert_eq(inspection.layout().length(), 12)
  assert_eq(inspection.layout().arena_bytes(), 768L)
}

///|
async test "readmission hashes exact bytes instead of trusting old file stamps" {
  let file = wb_stream_file()
  let inspection = wb_inspect_file(file)
  let current_stamp = wb_stamp(file, modified=17)
  let prepared = readmit_inspection(
    inspection,
    current_stamp,
    (position, count) => wb_slice(file, position, count),
    () => current_stamp,
  )
  assert_eq(prepared.layout, inspection.layout())
  assert_true(prepared.source_stamp == current_stamp)
}

///|
async test "readmission rejects changed content before device allocation" {
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
      (position, count) => wb_slice(changed, position, count),
      () => wb_stamp(changed, modified=19),
      _arena_bytes => {
        opens.val += 1
        1
      },
      (_resource, _bytes, _destination, _count) => (),
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
async test "ctime-only same-handle changes reject inspection admission" {
  let file = wb_stream_file()
  try
    prepare_streaming_load(
      wb_stream_model(file),
      wb_stamp(file, modified=7, changed=7),
      wb_stream_limits(),
      64L,
      4096L,
      (position, count) => wb_slice(file, position, count),
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
async test "readmission requires exact semantic transfer evidence" {
  let file = wb_stream_file()
  let admitted = wb_inspect_file(file)
  let altered : DeviceWeightFileInspection = {
    model: admitted.model,
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
      (position, count) => wb_slice(file, position, count),
      () => wb_stamp(file),
    )
  catch {
    FileChangedDuringLoad => ()
    _ => fail("unexpected evidence-mismatch error")
  } noraise {
    _ => fail("expected evidence-mismatch rejection")
  }
}
