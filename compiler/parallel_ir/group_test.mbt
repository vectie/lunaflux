///|
fn rank_program(
  rank : Int,
  kind? : @parallel_ir.CollectiveKind = SumAllReduce,
  element? : @parallel_ir.ElementType = BFloat16,
  receive_bytes? : Int64 = 4L,
) -> @parallel_ir.Program raise {
  @parallel_ir.compile(
    rank~,
    world_size=2,
    [@parallel_ir.Arena::new(bytes=16L, initialized=true)],
    [view(0L, 4L)],
    [
      @parallel_ir.Step::new(
        Communicate(
          @parallel_ir.Collective::new(
            sequence=1,
            kind~,
            element~,
            send_bytes=4L,
            receive_bytes~,
          ),
        ),
        [access(0, ReadWrite)],
      ),
    ],
  )
}

///|
test "group joins all ranks and rejects kind element geometry or rank mismatch" {
  let zero = rank_program(0)
  assert_eq(
    @parallel_ir.Group::new([zero, rank_program(1)]).ranks().length(),
    2,
  )
  for
    other in [
      rank_program(0),
      rank_program(1, element=Float32),
      rank_program(1, kind=LastAxisAllGather, receive_bytes=8L),
    ] {
    try @parallel_ir.Group::new([zero, other]) catch {
      _ => ()
    } noraise {
      _ => fail("incompatible group")
    }
  }
  try @parallel_ir.Group::new([zero]) catch {
    @parallel_ir.Group => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("missing rank")
  }
}

///|
test "collective order is independent of buffer hazards" {
  let program = compile(
    [
      collective(1, [access(0, ReadWrite)]),
      collective(2, [access(1, ReadWrite)]),
    ],
    [view(0L, 4L), view(8L, 4L)],
    initialized=true,
  )
  assert_true(program.happens_before(0, 1))
  try compile([collective(2, [])], []) catch {
    @parallel_ir.CollectiveOrder(1) => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("collective sequence gap")
  }
  try rank_program(0, receive_bytes=8L) catch {
    @parallel_ir.CollectiveOrder(1) => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("allreduce size mismatch")
  }
  try rank_program(0, kind=LastAxisAllGather, receive_bytes=4L) catch {
    @parallel_ir.CollectiveOrder(1) => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("gather size mismatch")
  }
}
