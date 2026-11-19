///|
async fn text_program() -> @text.Program {
  let model = @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
  let operators = @operators.DecoderOperators::new(model, maximum_rows=32)
  let attention = @attention.compile(
    operators,
    device_target=@catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=1,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
    target_id=@strategy.AttentionTargetId::new(
      "sm121-unqualified-software-profile",
    ),
    capabilities=@strategy.AttentionCompilerCapabilities::new(
      subgroup_width=32,
      max_threads_per_block=256,
      max_shared_memory_bytes=49152L,
      max_pipeline_stages=1,
      parallel_workgroup_target=1,
      supports_async_copy=false,
      matrix_shape=Some(
        @strategy.AttentionMatrixShape::new(rows=16, columns=16, reduction=16),
      ),
    ),
  )
  @text.Program::new(
    operators,
    @decoder.Program::new(operators, attention),
    maximum_workspace_bytes=1048576L,
  )
}

///|
async test "resident allocation budget covers weights storage and selected-row workspace" {
  let program = text_program()
  let budget = @minicpm_v47_runtime.Budget::for_text(
    program,
    weight_bytes=69319779584L,
  )
  assert_eq(budget.weight_bytes, 69319779584L)
  assert_eq(budget.storage_bytes, 152271612L)
  assert_eq(budget.output_bytes, 996672L)
  assert_eq(budget.total_allocation_bytes, 69473047868L)
  let mutable = @minicpm_v47_runtime.Budget::for_text(program, weight_bytes=0L)
  assert_eq(
    mutable.total_allocation_bytes,
    budget.storage_bytes + budget.output_bytes,
  )
  assert_true(
    try @minicpm_v47_runtime.Budget::for_text(program, weight_bytes=-1L) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_true(
    try
      @minicpm_v47_runtime.Budget::for_text(
        program,
        weight_bytes=9223372036854775807L,
      )
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
}

///|
async test "fresh execution has no request authority and closes without a device" {
  let execution = @minicpm_v47_runtime.Execution::new(text_program())
  assert_true(
    try execution.runner() catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  execution.close()
  execution.close()
  assert_true(
    try execution.runner() catch {
      _ => true
    } noraise {
      _ => false
    },
  )
}

///|
async test "AOT reader consumes exactly seven bounded modules and closes files on failure" {
  @async.with_task_group() <| group => {
    let directory = @fs.realpath(@fs.tmpdir(prefix="lunaflux-v47-aot-reader"))
    group.add_defer(() => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    })
    let root = @approved_fs.ApprovedRoot::open_absolute(directory)
    group.add_defer(() => root.close())
    let names = [
      "decoder_envelope", "recurrent_branch", "moe", "attention_common", "attention_prefill",
      "attention_decode", "text_io",
    ]
    for name in names {
      @fs.write_file(
        directory + "/" + name + ".cubin",
        b"AOT test bytes",
        create_mode=CreateNew,
      )
    }
    let images = @minicpm_v47_runtime.ModuleImages::read(
      root,
      maximum_module_bytes=64L,
    )
    for
      bytes in [
        images.envelope,
        images.recurrent,
        images.moe,
        images.attention_common,
        images.attention_prefill,
        images.attention_decode,
        images.text,
      ] {
      assert_eq(bytes, b"AOT test bytes")
    }
    assert_true(
      try
        @minicpm_v47_runtime.ModuleImages::read(root, maximum_module_bytes=4L)
      catch {
        _ => true
      } noraise {
        _ => false
      },
    )
    assert_true(
      try
        @minicpm_v47_runtime.ModuleImages::read(root, maximum_module_bytes=0L)
      catch {
        _ => true
      } noraise {
        _ => false
      },
    )
    @fs.write_file(directory + "/text_io.cubin", b"")
    assert_true(
      try
        @minicpm_v47_runtime.ModuleImages::read(root, maximum_module_bytes=64L)
      catch {
        _ => true
      } noraise {
        _ => false
      },
    )
    // Earlier modules were already read when the last file fails. Their file
    // handles must be closed so closing the approved root remains possible.
    root.close()
  }
}
