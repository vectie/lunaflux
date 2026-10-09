///|
fn access(
  buffer : Int,
  mode : @parallel_ir.AccessMode,
) -> @parallel_ir.Access raise {
  @parallel_ir.Access::new(buffer~, mode~)
}

///|
fn view(offset : Int64, bytes : Int64) -> @parallel_ir.Buffer raise {
  @parallel_ir.Buffer::new(arena=0, offset_bytes=offset, byte_count=bytes)
}

///|
fn compute(
  id : Int,
  accesses : ArrayView[@parallel_ir.Access],
  dependencies? : ArrayView[Int] = [],
) -> @parallel_ir.Step {
  @parallel_ir.Step::new(Compute(id), accesses, dependencies~)
}

///|
fn compile(
  steps : ArrayView[@parallel_ir.Step],
  buffers : ArrayView[@parallel_ir.Buffer],
  initialized? : Bool = false,
) -> @parallel_ir.Program raise {
  @parallel_ir.compile(
    rank=0,
    world_size=2,
    [@parallel_ir.Arena::new(bytes=64L, initialized~)],
    buffers,
    steps,
  )
}

///|
fn collective(
  sequence : Int,
  accesses : ArrayView[@parallel_ir.Access],
) -> @parallel_ir.Step raise {
  @parallel_ir.Step::new(
    Communicate(
      @parallel_ir.Collective::new(
        sequence~,
        kind=SumAllReduce,
        element=BFloat16,
        send_bytes=4L,
        receive_bytes=4L,
      ),
    ),
    accesses,
  )
}

///|
test "physical aliases infer RAW WAR WAW without serializing disjoint memory" {
  let program = compile(
    [
      compute(0, [access(0, Write)]),
      compute(1, [access(1, Read)]),
      compute(2, [access(2, Write)]),
      compute(3, [access(1, Write)]),
    ],
    [view(0L, 8L), view(4L, 4L), view(16L, 4L)],
  )
  assert_true(program.happens_before(0, 1))
  assert_true(program.happens_before(0, 3))
  assert_true(program.happens_before(1, 3))
  assert_false(program.happens_before(0, 2))
  assert_false(program.happens_before(2, 3))
  assert_eq(program.lifetime(1).unwrap().first_use(), Some(1))
  assert_eq(program.lifetime(1).unwrap().last_use(), Some(3))
}

///|
test "readers do not depend on each other and writer waits for every reader" {
  let program = compile(
    [
      compute(0, [access(0, Read)]),
      compute(1, [access(0, Read)]),
      compute(2, [access(0, Write)]),
    ],
    [view(0L, 4L)],
    initialized=true,
  )
  assert_false(program.happens_before(0, 1))
  assert_true(program.happens_before(0, 2))
  assert_true(program.happens_before(1, 2))
}

///|
test "partial overwrite retains dependencies on the untouched bytes" {
  let program = compile(
    [
      compute(0, [access(0, Write)]),
      compute(1, [access(1, Write)]),
      compute(2, [access(2, Read)]),
      compute(3, [access(0, Read)]),
    ],
    [view(0L, 12L), view(4L, 4L), view(8L, 4L)],
  )
  assert_true(program.happens_before(0, 2))
  assert_false(program.happens_before(1, 2))
  assert_true(program.happens_before(1, 3))
}

///|
test "temporary reads require complete initialization including gaps" {
  let buffers = [view(0L, 4L), view(8L, 4L), view(0L, 12L)]
  try
    compile(
      [
        compute(0, [access(0, Write), access(1, Write)]),
        compute(1, [access(2, Read)]),
      ],
      buffers,
    )
  catch {
    @parallel_ir.Uninitialized(step=1, buffer=2) => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("uninitialized gap accepted")
  }
  let valid = compile(
    [
      compute(0, [access(0, Write)]),
      compute(1, [access(1, Write)]),
      compute(2, [access(2, Read)]),
    ],
    [view(0L, 4L), view(4L, 4L), view(0L, 8L)],
  )
  assert_true(valid.happens_before(0, 2))
  assert_true(valid.happens_before(1, 2))
}

///|
test "same-step effects cannot invent input initialization or self edges" {
  try
    compile([compute(0, [access(0, Write), access(0, Read)])], [view(0L, 4L)])
  catch {
    @parallel_ir.Uninitialized(..) => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("same step initialized its own input")
  }
  let valid = compile(
    [compute(0, [access(0, Write), access(0, Read)])],
    [view(0L, 4L)],
    initialized=true,
  )
  assert_eq(valid.steps()[0].dependencies().length(), 0)
}

///|
test "explicit dependencies are topological and copied" {
  let dependencies = [0]
  let source = [compute(0, []), compute(1, [], dependencies~)]
  dependencies[0] = 1
  let program = compile(source, [])
  source[1] = compute(9, [])
  assert_true(program.happens_before(0, 1))
  try compile([compute(0, [], dependencies=[1]), compute(1, [])], []) catch {
    @parallel_ir.Dependency(0) => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("forward dependency/cycle accepted")
  }
}

///|
test "reuse waits for communication and all physical aliases" {
  let program = compile(
    [
      compute(0, [access(0, Write)]),
      collective(1, [access(0, ReadWrite)]),
      compute(1, []),
      compute(2, [access(1, Write)]),
    ],
    [view(0L, 4L), view(0L, 4L)],
  )
  assert_false(program.can_reuse_before(0, 2))
  assert_true(program.can_reuse_before(0, 3))
  assert_false(program.can_reuse_before(1, 2))
  assert_false(program.can_reuse_before(-1, 3))
}

///|
test "invalid rank buffer range and unknown buffer are rejected at compilation" {
  try
    @parallel_ir.Buffer::new(
      arena=0,
      offset_bytes=9223372036854775807L,
      byte_count=1L,
    )
  catch {
    @parallel_ir.Buffer(_) => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("overflow")
  }
  try compile([compute(0, [access(0, Write)])], [view(63L, 2L)]) catch {
    @parallel_ir.Buffer(_) => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("out of arena")
  }
  try compile([compute(0, [access(1, Write)])], [view(0L, 4L)]) catch {
    @parallel_ir.Buffer(_) => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("unknown buffer")
  }
  try
    @parallel_ir.compile(
      rank=2,
      world_size=2,
      [@parallel_ir.Arena::new(bytes=1L, initialized=true)],
      [],
      [compute(0, [])],
    )
  catch {
    @parallel_ir.Capacity => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("invalid rank")
  }
}
