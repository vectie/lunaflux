///|
async fn luna_tile_probe_write_new(path : String, bytes : Bytes) -> Unit {
  @fs.write_file(
    path,
    bytes,
    create_mode=CreateNew,
    permission=0o600,
    sync=Full,
  )
}

///|
async fn export_luna_tile_parallel_probe(args : Array[String]) -> Unit {
  require_luna_tile_parallel_probe(
    args.length() == 3,
    "usage: luna_tile_parallel_cuda_probe export ABSOLUTE_EMPTY_OUTPUT",
  )
  let output = args[2]
  require_luna_tile_parallel_probe(
    output.has_prefix("/") && @fs.exists(output),
    "export output must be an existing absolute directory",
  )
  let (serial, parallel) = luna_tile_probe_candidates()
  let record = luna_tile_probe_fixture_record(serial, parallel)
  luna_tile_probe_write_new(output + "/serial.cu", serial.source_bytes())
  luna_tile_probe_write_new(
    output + "/serial.canonical",
    serial.canonical_bytes(),
  )
  luna_tile_probe_write_new(output + "/parallel.cu", parallel.source_bytes())
  luna_tile_probe_write_new(
    output + "/parallel.canonical",
    parallel.canonical_bytes(),
  )
  luna_tile_probe_write_new(output + "/fixture.v1", record)
  println(
    "outcome=lunatile-parallel-sm120-fixture-exported fixture_sha256=\{luna_tile_probe_sha256(record)} serial_source_sha256=\{serial.source_digest().as_hex()} parallel_source_sha256=\{parallel.source_digest().as_hex()} authority=qualification-only",
  )
}
