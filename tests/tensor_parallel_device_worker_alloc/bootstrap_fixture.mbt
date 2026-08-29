///|
priv struct WorkerFixture {
  workers : Array[
    @tensor_parallel_device_worker.TensorParallelDeviceWorkerOwner,
  ]
  operation_count : Int
  collective_count : Int
}

///|
fn startup_contract(
  fixture : ExecutionFixture,
  rank : Int,
) -> @worker_wire.WorkerStartupContract raise {
  @worker_wire.WorkerStartupContract::new(
    identity=fixture.model.identity(),
    bootstrap_digest=@worker_wire.WorkerBootstrapDigest::from_sha256(
      if rank == 0 {
        "1111111111111111111111111111111111111111111111111111111111111111"
      } else {
        "2222222222222222222222222222222222222222222222222222222222222222"
      },
    ),
    bootstrap_source_digest=fixture.topology.source.digest(),
    device_ordinal=rank,
    model_generation_value=TP_MODEL_GENERATION,
    worker_limits=fixture_worker_limits(),
    inference_limits=fixture_inference_limits(),
    has_predecessor=false,
    predecessor_value=0UL,
  )
}

///|
fn wait_for_worker_ready(
  worker : @tensor_parallel_device_worker.TensorParallelDeviceWorkerOwner,
) -> Unit raise {
  let mut polls = 0
  while !worker.poll_startup() {
    polls += 1
    if polls > 4 {
      abort("fake collective startup did not become ready")
    }
  }
  if polls != 1 || worker.lifecycle() != WorkerReady {
    abort("startup did not prove false-then-true polling")
  }
  ignore(worker.readiness_contract())
}

///|
fn prepare_workers(
  root : @approved_fs.ApprovedRoot,
  fixture : ExecutionFixture,
) -> WorkerFixture raise {
  let model_generation = @worker_protocol.ModelPlanGeneration::from_wire(
    TP_MODEL_GENERATION,
  )
  let group_generation = @rank_group_protocol.RankGroupGeneration::from_wire(
    TP_GROUP_GENERATION,
  )
  let startups : Array[@worker_wire.WorkerStartupContract] = []
  let executions : Array[
    @tensor_parallel_execution_plan.TensorParallelExecutionPlan,
  ] = []
  for rank in 0..<TP_WORLD_SIZE {
    startups.push(startup_contract(fixture, rank))
    executions.push(fixture.ranks[rank].execution)
  }
  let group = @tensor_parallel_worker_bootstrap.admit(
    fixture.model,
    model_generation,
    group_generation,
    fixture.topology.topology,
    fixture.topology.source,
    startups,
    fixture.inspections,
    fixture.devices,
    executions,
    fixture.kv_plan,
    fixture.topology.rendezvous,
    collective_runtime_version_code=TP_RUNTIME_VERSION,
  )
  let workers : Array[
    @tensor_parallel_device_worker.TensorParallelDeviceWorkerOwner,
  ] = []
  for rank in 0..<TP_WORLD_SIZE {
    guard group.rank_envelope(rank) is Some(envelope) else {
      abort("rank bootstrap envelope is missing")
    }
    let rank_execution = fixture.ranks[rank]
    let admitted = @tensor_parallel_device_worker.admit_plan(
      fixture.model,
      model_generation,
      group_generation,
      fixture.topology.topology,
      startups[rank],
      envelope,
      fixture.inspections[rank],
      fixture.devices[rank],
      rank_execution.launch_set,
      rank_execution.artifacts,
      fixture.profile,
      fixture.execution_limits,
      fixture.kv_plan,
      group.digest(),
      fixture.topology.rendezvous,
      collective_runtime_version_code=TP_RUNTIME_VERSION,
    )
    let outcome = @tensor_parallel_device_worker.prepare(
      root,
      fixture.topology.runtime,
      admitted,
    )
    match outcome {
      TensorParallelWorkerPrepared(worker) => {
        wait_for_worker_ready(worker)
        workers.push(worker)
      }
      TensorParallelWorkerCleanupRequired(failed) => {
        try! failed.retry_close()
        abort("TP worker preparation returned cleanup authority")
      }
    }
  }
  {
    workers,
    operation_count: fixture.ranks[0].execution.operation_count(),
    collective_count: fixture.ranks[0].execution.collective_count(),
  }
}

///|
fn close_workers(fixture : WorkerFixture) -> Unit raise {
  for worker in fixture.workers {
    worker.begin_drain()
    worker.close()
    if worker.lifecycle() != WorkerClosed {
      abort("TP worker did not close")
    }
  }
}
