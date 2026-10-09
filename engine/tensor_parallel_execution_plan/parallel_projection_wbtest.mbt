///|
async test "parallel IR projects admitted TP kernels collectives and physical aliases" {
  @async.with_task_group() <| tasks => {
    let directory = @fs.tmpdir(prefix="lunaflux-parallel-ir")
    tasks.add_defer(() => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    })
    let file = exec_weight_file()
    @fs.write_file(
      "\{directory}/weights.safetensors",
      file,
      create_mode=CreateNew,
    )
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    defer (root.close() catch { _ => () })
    let plans = [0, 1].map(rank => {
      let source = build_execution_fixture(root, file, rank, "parallel-module")
      admit(
        source.model,
        source.launch_set,
        source.device_plan,
        source.artifacts,
        source.profiles[0],
        exec_limits(),
      )
    })
    let group = parallel_group(plans)
    assert_eq(group.ranks().length(), 2)
    for rank, program in group.ranks() {
      let physical = plans[rank]
      assert_eq(program.rank(), rank)
      assert_eq(
        program.steps().length(),
        physical.operation_count() + physical.collective_count(),
      )
      assert_eq(
        program.collective_steps().length(),
        physical.collective_count(),
      )
      let ordered = program.lower(Ordered)
      assert_eq(
        ordered.commands(ComputeLane).length(),
        program.steps().length(),
      )
      assert_eq(ordered.event_producers().length(), 0)
      let overlap = program.lower(ComputeCommunicationOverlap)
      assert_true(overlap.event_producers().length() > 0)
      for index, step in program.collective_steps() {
        guard program.steps()[step].action() is Communicate(collective) else {
          fail("missing collective")
        }
        assert_eq(collective.sequence(), index + 1)
        assert_eq(
          collective.receive_bytes(),
          physical.collectives[index].receive_bytes,
        )
        assert_true(program.happens_before(step - 1, step))
      }
      // Opt-in projection does not change the admitted execution identity.
      let digest = physical.digest()
      ignore(physical.parallel_program())
      assert_eq(physical.digest(), digest)
    }
    try parallel_group([plans[1], plans[0]]) catch {
      _ => ()
    } noraise {
      _ => fail("rank reorder accepted")
    }
  }
}
