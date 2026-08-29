///|
fn converter_append_u64(output : Array[Byte], value : UInt64) -> Unit {
  for byte in value.to_le_bytes() {
    output.push(byte)
  }
}

///|
fn converter_append_bytes(output : Array[Byte], bytes : BytesView) -> Unit {
  for byte in bytes {
    output.push(byte)
  }
}

///|
async test "native converter copies only ordered regions and refuses overwrite" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-qwen3-convert")
    let root = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root, recursive=true))
    }
    let source = root + "/source.safetensors"
    let output = root + "/numeric.safetensors"
    let source_bytes : Array[Byte] = []
    converter_append_u64(source_bytes, 8UL)
    converter_append_bytes(source_bytes, @utf8.encode("{}      "))
    converter_append_bytes(source_bytes, b"abcdefgh")
    @fs.write_file(
      source,
      Bytes::from_array(source_bytes),
      create_mode=CreateNew,
    )
    let regions : Array[Byte] = []
    converter_append_u64(regions, 2UL)
    converter_append_u64(regions, 3UL)
    converter_append_u64(regions, 0UL)
    converter_append_u64(regions, 2UL)
    let header = @utf8.encode("{}      ")
    assert_eq(
      raw_convert_qwen3_numeric_file(
        @utf8.encode(source),
        @utf8.encode(output),
        header,
        Bytes::from_array(regions),
        2,
      ),
      0,
    )
    let expected : Array[Byte] = []
    converter_append_u64(expected, 8UL)
    converter_append_bytes(expected, header)
    converter_append_bytes(expected, b"cdeab")
    assert_eq(@fs.read_file(output).binary(), Bytes::from_array(expected))
    assert_eq(
      raw_convert_qwen3_numeric_file(
        @utf8.encode(source),
        @utf8.encode(output),
        header,
        Bytes::from_array(regions),
        2,
      ),
      3,
    )
    assert_eq(@fs.read_file(output).binary(), Bytes::from_array(expected))
  }
}

///|
async test "native converter rejects truncated paths and removes partial output" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-qwen3-convert-hostile")
    let root = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root, recursive=true))
    }
    let source = root + "/source.safetensors"
    let output = root + "/partial.numeric"
    let source_bytes : Array[Byte] = []
    converter_append_u64(source_bytes, 8UL)
    converter_append_bytes(source_bytes, @utf8.encode("{}      "))
    converter_append_bytes(source_bytes, b"ab")
    @fs.write_file(
      source,
      Bytes::from_array(source_bytes),
      create_mode=CreateNew,
    )
    let regions : Array[Byte] = []
    converter_append_u64(regions, 0UL)
    converter_append_u64(regions, 3UL)
    assert_eq(
      raw_convert_qwen3_numeric_file(
        @utf8.encode(source),
        @utf8.encode(output),
        @utf8.encode("{}      "),
        Bytes::from_array(regions),
        2,
      ),
      4,
    )
    assert_false(@fs.exists(output))
    let hostile_path = @utf8.encode(source + "\u0000suffix")
    assert_eq(
      raw_convert_qwen3_numeric_file(
        hostile_path,
        @utf8.encode(output),
        @utf8.encode("{}      "),
        Bytes::from_array(regions),
        2,
      ),
      1,
    )
    assert_false(@fs.exists(output))
  }
}
