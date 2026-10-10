///|
test "full request export keeps each argument in its semantic role" {
  let options = request_export_arguments([
    "32", "48", "120", "5", "3", "256", "64", "4096", "8589934592", "/tmp/export",
  ])
  assert_eq((options.height, options.width, options.frames), (32, 48, 120))
  assert_eq((options.steps, options.text_rows), (5, 3))
  assert_eq(
    (options.tile_size, options.overlap, options.max_tiles),
    (256, 64, 4096),
  )
  assert_eq(options.workspace_bytes, 8589934592L)
  assert_eq(options.output_directory, "/tmp/export")
}

///|
test "full request export rejects empty or invalid input before generating source" {
  let base = [
    "32", "32", "120", "5", "3", "256", "64", "4096", "8589934592", "/tmp/export",
  ]
  for
    (index, value) in [
      (0, "0"),
      (1, "-1"),
      (2, "0"),
      (3, "0"),
      (4, "0"),
      (5, "0"),
      (6, "256"),
      (6, "-1"),
      (7, "0"),
      (8, "0"),
      (8, "9223372036854775808"),
      (9, ""),
    ] {
    let input = base.copy()
    input[index] = value
    try request_export_arguments(input[:]) catch {
      _ => ()
    } noraise {
      _ => fail("invalid export input accepted")
    }
  }
  try request_export_arguments(base[:9]) catch {
    _ => ()
  } noraise {
    _ => fail("missing export argument accepted")
  }
}

///|
test "request source row capacity comes from hidden and index storage budgets" {
  assert_eq(request_source_row_limit(1000L, 1000L, 10), 50)
  assert_eq(request_source_row_limit(1000L, 80L, 10), 10)
  assert_true(
    request_source_row_limit(8589934592L, 1073741824L, 5376) < 268435455,
  )
  for
    (workspace, metadata, width) in [
      (0L, 80L, 10),
      (100L, 7L, 10),
      (100L, 80L, 0),
      (100L, 2147483648L, 10),
    ] {
    try request_source_row_limit(workspace, metadata, width) catch {
      _ => ()
    } noraise {
      _ => fail("invalid source capacity accepted")
    }
  }
}
