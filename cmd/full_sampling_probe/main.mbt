///|
async fn main {
  guard @env.args() is [_, mode, path, vocabulary, maximum_rows] else {
    fail("export|run FILE VOCABULARY MAXIMUM_ROWS")
  }
  let v = @string.parse_int(vocabulary)
  let rows = @string.parse_int(maximum_rows)
  guard v >= 9 && v <= 262144 && rows >= 1 && rows <= 8 else {
    fail("fixture geometry")
  }
  let program = @physical.FullVocabularySamplingProgram::new(
    values=v,
    maximum_rows=rows,
    maximum_workspace_bytes=67108864L,
  ).unwrap()
  match mode {
    "export" =>
      @fs.write_file(
        path,
        @utf8.encode(@source.source(program)),
        create_mode=CreateNew,
      )
    "run" => {
      let context = @device.open_context(0)
      defer (context.close() catch { _ => () })
      let module = context.load_module(@fs.read_file(path).binary())
      defer (module.close() catch { _ => () })
      let logits = context.allocate(v.to_int64() * rows.to_int64() * 4L)
      defer (logits.close() catch { _ => () })
      let sampler = @gpu.BatchedDeviceSampler::new(
        context,
        module,
        logits,
        program,
        maximum_device_bytes=71303168L,
        maximum_host_bytes=1048576L,
      )
      defer (sampler.close() catch { _ => () })
      run_cases(context, logits, sampler, v, rows)
      sampler.close()
      sampler.close()
      logits.close()
      module.close()
      context.close()
      println("full_sampling_actual_owner=pass context_closed=true")
    }
    _ => fail("export|run")
  }
}
