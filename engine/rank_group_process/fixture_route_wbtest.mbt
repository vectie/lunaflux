///|
fn prepare_rank_group_process(
  executable : Bytes,
  rank_startups : ArrayView[RankGroupProcessRankStartup],
  plan_buffer_identities : ArrayView[@worker_protocol.PlanBufferIdentity],
  previous_plan_sequence_value~ : UInt64,
  limits : RankGroupProcessLimits,
) -> PreparedRankGroupProcess raise RankGroupProcessError {
  let executable = @worker_executable_fixture.admit(executable) catch {
    _ => raise Invalid(Limit)
  }
  prepare_rank_group_process_approved(
    executable,
    rank_startups,
    plan_buffer_identities,
    previous_plan_sequence_value~,
    limits,
  )
}

///|
fn spawn_rank_fixture_child(
  path : Bytes,
  limits : @process.ProcessLimits,
) -> @process.ChildProcess raise {
  let executable = @worker_executable_fixture.admit(path)
  let model = @approved_fs.ApprovedRoot::open_absolute("/usr")
  let kernel = @approved_fs.ApprovedRoot::open_absolute("/usr")
  let roots = @approved_fs.acquire_worker_approved_roots(
    model_root=model,
    kernel_root=kernel,
  )
  let child = executable
    .prepare_child(limits)
    .spawn_with_approved_roots(
      @approved_fs.worker_approved_root_spawn_authority(roots),
    )
  @approved_fs.close_worker_approved_roots(roots)
  model.close()
  kernel.close()
  executable.close()
  child
}
