///|
fn network_test_topology() -> @network_topology.TwoNodeTopology raise {
  let target = tp_bootstrap_target()
  let ranks = ["spark-a", "spark-b"].map(node => {
    @network_topology.NetworkRank::new(
      node=@network_topology.NodeIdentity::new(node),
      device_ordinal=0,
      target~,
      physical_memory_bytes=8388608L,
      rdma_interface="mlx5_0",
    )
  })
  @network_topology.TwoNodeTopology::new(ranks[0], ranks[1])
}

///|
fn network_test_startup(
  identity : @spec.ModelIdentity,
) -> @worker_wire.WorkerStartupContract raise {
  @worker_wire.WorkerStartupContract::new(
    identity~,
    bootstrap_digest=@worker_wire.WorkerBootstrapDigest::from_sha256(
      "1".repeat(64),
    ),
    bootstrap_source_digest=tp_bootstrap_source().digest(),
    device_ordinal=0,
    model_generation_value=7UL,
    has_predecessor=false,
    predecessor_value=0UL,
    worker_limits=@worker_protocol.WorkerProtocolLimits::new(
      max_prefill_rows=2,
      max_decode_rows=2,
      max_plan_rows=2,
      max_plan_tokens=16,
      max_plan_pages=2,
      max_capabilities=32,
      max_completion_slots=2,
      max_sequence_tokens=16,
      max_token_id=7,
    ),
    inference_limits=@inference.InferenceLimits::new(
      max_text_bytes=128,
      max_input_tokens=8,
      max_new_tokens=8,
      max_context_tokens=16,
      max_token_id=7,
      max_stop_token_ids=2,
      max_stop_strings=2,
      max_stop_string_bytes=16,
      max_trace_bytes=16,
      max_cache_scope_bytes=16,
      max_decoded_delta_bytes=16,
      max_deadline_millis=1000UL,
      max_top_k=8,
      max_temperature=4.0,
    ),
  )
}

///|
async test "two independent ordinal-zero nodes join sharded files AOT plans and the existing worker" {
  @async.with_task_group() <| tasks => {
    let directory = @fs.tmpdir(prefix="lunaflux-network-bootstrap")
    tasks.add_defer(() => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    })
    let file = tp_bootstrap_weight_file()
    @fs.write_file(
      "\{directory}/weights.safetensors",
      file,
      create_mode=CreateNew,
    )
    let metadata = tp_bootstrap_model(file)
    let model = @llama.build_paged(
      metadata,
      batch=@llama.LlamaPagedBatchEnvelope::new(2),
    )
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    defer (root.close() catch { _ => () })
    let inspections = tp_bootstrap_inspections(root, metadata, model)
    let topology = network_test_topology()
    let budget = @two_node_plan.RankMemoryBudget::new(
      system_reserved_bytes=1048576L,
      runtime_ceiling_bytes=4194304L,
      activation_workspace_bytes=2097152L,
      collective_bytes=1048576L,
      kv_bytes=4096L,
    )
    let compiled = @two_node_plan.compile(
      model,
      topology,
      [inspections[0].rank_plan(), inspections[1].rank_plan()],
      [budget, budget],
      tokens_per_page=128,
      maximum_pages=4,
    )
    let devices = [0, 1].map(rank => {
      @tensor_parallel_device_plan.build(
        model,
        inspections[rank].rank_plan(),
        inspections[rank],
        @tensor_parallel_device_plan.TensorParallelRankDevice::from_network_topology(
          topology,
          rank~,
        ),
        topology.target(),
      )
    })
    let kv = @tensor_parallel_kv_plan.admit_from_two_node_plan(
      model,
      @worker_protocol.ModelPlanGeneration::from_wire(7UL),
      compiled,
    )
    let executions = tp_bootstrap_execution_plans(
      model, inspections, devices, kv,
    )
    let startup = network_test_startup(model.identity())
    let executable = @spec.ContentDigest::from_sha256("a".repeat(64))
    let remote = @remote_startup.RemoteGroupStartup::new(
      compiled,
      [startup, startup],
      [executable, executable],
      deployment_digest=executable,
      group_generation=9UL,
      rendezvous=bootstrap_test_rendezvous(),
      lease_millis=1000UL,
    )
    let admitted = admit_network(
      model,
      compiled,
      remote,
      inspections,
      devices,
      executions,
      kv,
      collective_runtime_version_code=21403,
    )
    exercise_remote_group(remote, admitted) catch {
      error => fail("remote protocol: \{Repr(error)}")
    }
    exercise_remote_tls_group(remote, admitted)
    exercise_remote_group(remote, admitted, disconnect_after_accept=true)
    exercise_remote_service(remote, admitted, model)
    let replacements = [10UL, 11UL].map(generation => {
      @remote_startup.RemoteGroupStartup::new(
        compiled,
        [startup, startup],
        [executable, executable],
        deployment_digest=executable,
        group_generation=generation,
        rendezvous=bootstrap_test_rendezvous(),
        lease_millis=1000UL,
      )
    })
    let replacement_admissions = replacements.map(replacement => {
      admit_network(
        model,
        compiled,
        replacement,
        inspections,
        devices,
        executions,
        kv,
        collective_runtime_version_code=21403,
      )
    })
    exercise_remote_recovery(
      model,
      remote,
      admitted,
      replacements[0],
      replacement_admissions[0],
      replacements[1],
      replacement_admissions[1],
    )
    let changed_lease = @remote_startup.RemoteGroupStartup::new(
      compiled,
      [startup, startup],
      [executable, executable],
      deployment_digest=executable,
      group_generation=9UL,
      rendezvous=bootstrap_test_rendezvous(),
      lease_millis=2000UL,
    )
    try
      admitted
      .rank_envelope(0)
      .unwrap()
      .authenticate_network(
        model,
        compiled,
        changed_lease.rank(0).unwrap(),
        inspections[0],
        devices[0],
        executions[0],
        kv,
        expected_group_digest=admitted.digest(),
        expected_collective_runtime_version_code=21403,
      )
    catch {
      Rejected(rule=Topology, ..) => ()
      error => fail("unexpected: \{Repr(error)}")
    } noraise {
      _ => fail("changed remote lease accepted under old envelope")
    }
    let module_bytes = @utf8.encode("tp-bootstrap-module-primary")
    let module_digest = @catalog.AotArtifactDigest::from_sha256(
      @crypto.bytes_to_hex_string(@crypto.sha256(module_bytes)),
    )
    let catalog = tp_execution_catalog(model, topology.target(), module_digest)
    for rank in 0..<2 {
      let envelope = tp_bootstrap_decode(admitted.rank_envelope(rank).unwrap())
      let contract = envelope.authenticate_network(
        model,
        compiled,
        remote.rank(rank).unwrap(),
        inspections[rank],
        devices[rank],
        executions[rank],
        kv,
        expected_group_digest=admitted.digest(),
        expected_collective_runtime_version_code=21403,
      )
      assert_eq(contract.rank(), rank)
      assert_eq(contract.world_size(), 2)
      let launch_set = @tensor_parallel_launch_contract.admit_paged_rank(
        model,
        inspections[rank].rank_plan(),
        kv.rank_layout(rank),
        catalog,
        expected_rank=rank,
        expected_world_size=2,
        expected_target=topology.target(),
        [tp_execution_profile()],
        tp_execution_launch_inputs(
          model,
          inspections[rank].rank_plan(),
          topology.target(),
          module_digest,
          rank,
        ),
        @tensor_parallel_launch_contract.TensorParallelLaunchContractLimits::new(
          max_profiles=2,
          max_contracts=64,
          max_operands_per_contract=16,
        ),
      )
      ignore(
        @tensor_parallel_device_worker.admit_network_plan(
          model,
          compiled,
          remote.rank(rank).unwrap(),
          envelope,
          inspections[rank],
          devices[rank],
          launch_set,
          tp_execution_artifacts(launch_set, module_bytes, module_digest),
          tp_execution_profile(),
          tp_execution_limits(),
          kv,
          admitted.digest(),
          collective_runtime_version_code=21403,
        ),
      )
      for foreign_rank in [1 - rank] {
        try
          envelope.authenticate_network(
            model,
            compiled,
            remote.rank(foreign_rank).unwrap(),
            inspections[foreign_rank],
            devices[foreign_rank],
            executions[foreign_rank],
            kv,
            expected_group_digest=admitted.digest(),
            expected_collective_runtime_version_code=21403,
          )
        catch {
          Rejected(rule=Topology, ..) => ()
          error => fail("unexpected: \{Repr(error)}")
        } noraise {
          _ => fail("rank substitution accepted")
        }
      }
    }
    let alternate = tp_bootstrap_execution_plans(
      model,
      inspections,
      devices,
      kv,
      module_text="other-artifact",
    )
    try
      admitted
      .rank_envelope(0)
      .unwrap()
      .authenticate_network(
        model,
        compiled,
        remote.rank(0).unwrap(),
        inspections[0],
        devices[0],
        alternate[0],
        kv,
        expected_group_digest=admitted.digest(),
        expected_collective_runtime_version_code=21403,
      )
    catch {
      Rejected(rule=Digest, ..) => ()
      error => fail("unexpected: \{Repr(error)}")
    } noraise {
      _ => fail("artifact substitution accepted")
    }
  }
}
