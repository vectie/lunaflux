///|
async test "public group admit authenticates execution plans and rejects worker or launch-artifact substitution" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-tp-worker-bootstrap")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let file = tp_bootstrap_weight_file()
    @fs.write_file(
      "\{directory}/weights.safetensors",
      file,
      create_mode=CreateNew,
    )
    let model = tp_bootstrap_model(file)
    let model_plan = @llama.build_paged(
      model,
      batch=@llama.LlamaPagedBatchEnvelope::new(2),
    )
    let topology = tp_bootstrap_topology()
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    let inspections = tp_bootstrap_inspections(root, model, model_plan)
    let device_plans = tp_bootstrap_device_plans(
      topology, model_plan, inspections,
    )
    let generation = @worker_protocol.ModelPlanGeneration::from_wire(7UL)
    let group_generation = @rank_group_protocol.RankGroupGeneration::from_wire(
      9UL,
    )
    let runtime = tp_bootstrap_runtime(topology)
    let kv_plan = @tensor_parallel_kv_plan.admit(
      model_plan,
      generation,
      runtime,
      device_plans,
      tokens_per_page=128,
      total_page_count=4,
    )
    let execution_plans = tp_bootstrap_execution_plans(
      model_plan, inspections, device_plans, kv_plan,
    )
    let startups = [
      tp_bootstrap_startup(model_plan.identity(), 0),
      tp_bootstrap_startup(model_plan.identity(), 1),
    ]
    let rendezvous = bootstrap_test_rendezvous()
    let admitted = admit(
      model_plan,
      generation,
      group_generation,
      topology,
      tp_bootstrap_source(),
      startups,
      inspections,
      device_plans,
      execution_plans,
      kv_plan,
      rendezvous,
      collective_runtime_version_code=21403,
    )
    assert_eq(admitted.world_size(), 2)
    assert_eq(
      admitted.template_digest_sha256(),
      "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    )
    let alternate_template_group = admit_with_template(
      "8888888888888888888888888888888888888888888888888888888888888888",
      model_plan,
      generation,
      group_generation,
      topology,
      tp_bootstrap_source(),
      startups,
      inspections,
      device_plans,
      execution_plans,
      kv_plan,
      rendezvous,
      collective_runtime_version_code=21403,
    )
    assert_true(alternate_template_group.digest() != admitted.digest())
    assert_eq(
      alternate_template_group.template_digest_sha256(),
      "8888888888888888888888888888888888888888888888888888888888888888",
    )
    for rank in 0..<2 {
      let decoded = tp_bootstrap_decode(admitted.rank_envelope(rank).unwrap())
      let contract = decoded.authenticate_local(
        model_plan,
        generation,
        group_generation,
        topology,
        startups[rank],
        inspections[rank],
        device_plans[rank],
        execution_plans[rank],
        kv_plan.rank_binding(rank),
        expected_group_digest=admitted.digest(),
        expected_rendezvous_id=rendezvous,
        expected_collective_runtime_version_code=21403,
      )
      let expected = @tensor_parallel_collective.authenticate_rank_plan(
        inspections[rank].rank_plan(),
        expected_identity=model_plan.identity(),
        max_query_tokens=startups[rank].worker_limits().max_plan_tokens(),
      )
      assert_eq(contract.digest(), expected.digest())
      assert_eq(contract.rank(), rank)
      assert_eq(contract.world_size(), 2)
      assert_eq(contract.site_count(), expected.site_count())
    }
    let decoded = tp_bootstrap_decode(admitted.rank_envelope(0).unwrap())
    let other_rendezvous = bootstrap_test_rendezvous(fill=b'\x6b')
    let other_group = admit(
      model_plan,
      generation,
      group_generation,
      topology,
      tp_bootstrap_source(),
      startups,
      inspections,
      device_plans,
      execution_plans,
      kv_plan,
      other_rendezvous,
      collective_runtime_version_code=21403,
    )
    let alternate_execution_plans = tp_bootstrap_execution_plans(
      model_plan,
      inspections,
      device_plans,
      kv_plan,
      module_text="tp-bootstrap-module-alternate",
    )
    let alternate_execution_group = admit(
      model_plan,
      generation,
      group_generation,
      topology,
      tp_bootstrap_source(),
      startups,
      inspections,
      device_plans,
      alternate_execution_plans,
      kv_plan,
      rendezvous,
      collective_runtime_version_code=21403,
    )
    assert_true(alternate_execution_group.digest() != admitted.digest())
    assert_true(
      alternate_execution_group
      .rank_envelope(0)
      .unwrap()
      .execution_plan_digest_sha256() !=
      admitted.rank_envelope(0).unwrap().execution_plan_digest_sha256(),
    )
    tp_bootstrap_expect_local_rule(
      () => {
        ignore(
          decoded.authenticate_local(
            model_plan,
            generation,
            @rank_group_protocol.RankGroupGeneration::from_wire(10UL),
            topology,
            startups[0],
            inspections[0],
            device_plans[0],
            execution_plans[0],
            kv_plan.rank_binding(0),
            expected_group_digest=admitted.digest(),
            expected_rendezvous_id=rendezvous,
            expected_collective_runtime_version_code=21403,
          ),
        )
      },
      GroupGeneration,
    )
    tp_bootstrap_expect_local_rule(
      () => {
        ignore(
          decoded.authenticate_local(
            model_plan,
            generation,
            group_generation,
            topology,
            startups[0],
            inspections[0],
            device_plans[0],
            execution_plans[0],
            kv_plan.rank_binding(0),
            expected_group_digest=other_group.digest(),
            expected_rendezvous_id=rendezvous,
            expected_collective_runtime_version_code=21403,
          ),
        )
      },
      Digest,
    )
    tp_bootstrap_expect_local_rule(
      () => {
        ignore(
          decoded.authenticate_local(
            model_plan,
            generation,
            group_generation,
            topology,
            startups[0],
            inspections[0],
            device_plans[0],
            alternate_execution_plans[0],
            kv_plan.rank_binding(0),
            expected_group_digest=admitted.digest(),
            expected_rendezvous_id=rendezvous,
            expected_collective_runtime_version_code=21403,
          ),
        )
      },
      ExecutionPlan,
    )
    tp_bootstrap_expect_local_rule(
      () => {
        ignore(
          decoded.authenticate_local(
            model_plan,
            generation,
            group_generation,
            topology,
            startups[0],
            inspections[0],
            device_plans[0],
            execution_plans[0],
            kv_plan.rank_binding(0),
            expected_group_digest=admitted.digest(),
            expected_rendezvous_id=other_rendezvous,
            expected_collective_runtime_version_code=21403,
          ),
        )
      },
      RendezvousId,
    )
    tp_bootstrap_expect_local_rule(
      () => {
        ignore(
          decoded.authenticate_local(
            model_plan,
            generation,
            group_generation,
            topology,
            startups[0],
            inspections[0],
            device_plans[0],
            execution_plans[0],
            kv_plan.rank_binding(1),
            expected_group_digest=admitted.digest(),
            expected_rendezvous_id=rendezvous,
            expected_collective_runtime_version_code=21403,
          ),
        )
      },
      KvContract,
    )
    tp_bootstrap_expect_local_rule(
      () => {
        ignore(
          decoded.authenticate_local(
            model_plan,
            generation,
            group_generation,
            topology,
            startups[1],
            inspections[1],
            device_plans[1],
            execution_plans[1],
            kv_plan.rank_binding(1),
            expected_group_digest=admitted.digest(),
            expected_rendezvous_id=rendezvous,
            expected_collective_runtime_version_code=21403,
          ),
        )
      },
      DeviceOrdinal,
    )
    let substituted_worker = tp_bootstrap_startup(
      model_plan.identity(),
      0,
      worker_limits=tp_bootstrap_worker_limits_with_plan_tokens(31),
    )
    tp_bootstrap_expect_local_rule(
      () => {
        ignore(
          decoded.authenticate_local(
            model_plan,
            generation,
            group_generation,
            topology,
            substituted_worker,
            inspections[0],
            device_plans[0],
            execution_plans[0],
            kv_plan.rank_binding(0),
            expected_group_digest=admitted.digest(),
            expected_rendezvous_id=rendezvous,
            expected_collective_runtime_version_code=21403,
          ),
        )
      },
      WorkerContract,
    )
    root.close()
  }
}

///|
async test "public admission rejects mixed worker sequence and query-limit domains" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-tp-worker-sequence")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let file = tp_bootstrap_weight_file()
    @fs.write_file(
      "\{directory}/weights.safetensors",
      file,
      create_mode=CreateNew,
    )
    let model = tp_bootstrap_model(file)
    let model_plan = @llama.build_paged(
      model,
      batch=@llama.LlamaPagedBatchEnvelope::new(2),
    )
    let topology = tp_bootstrap_topology()
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    let inspections = tp_bootstrap_inspections(root, model, model_plan)
    let device_plans = tp_bootstrap_device_plans(
      topology, model_plan, inspections,
    )
    let generation = @worker_protocol.ModelPlanGeneration::from_wire(7UL)
    let kv_plan = @tensor_parallel_kv_plan.admit(
      model_plan,
      generation,
      tp_bootstrap_runtime(topology),
      device_plans,
      tokens_per_page=128,
      total_page_count=4,
    )
    let execution_plans = tp_bootstrap_execution_plans(
      model_plan, inspections, device_plans, kv_plan,
    )
    let group_generation = @rank_group_protocol.RankGroupGeneration::from_wire(
      9UL,
    )
    for
      startups in [
        [
          tp_bootstrap_startup(model_plan.identity(), 0),
          tp_bootstrap_startup(
            model_plan.identity(),
            1,
            has_predecessor=true,
            predecessor_value=1UL,
          ),
        ],
        [
          tp_bootstrap_startup(
            model_plan.identity(),
            0,
            has_predecessor=true,
            predecessor_value=1UL,
          ),
          tp_bootstrap_startup(
            model_plan.identity(),
            1,
            has_predecessor=true,
            predecessor_value=2UL,
          ),
        ],
        [
          tp_bootstrap_startup(model_plan.identity(), 0),
          tp_bootstrap_startup(
            model_plan.identity(),
            1,
            worker_limits=tp_bootstrap_worker_limits_with_plan_tokens(31),
          ),
        ],
        [
          tp_bootstrap_startup(model_plan.identity(), 0),
          tp_bootstrap_startup(
            model_plan.identity(),
            1,
            source=tp_bootstrap_source(
              maximum_collective_runtime_version_code=21404,
            ),
          ),
        ],
      ] {
      try
        ignore(
          admit(
            model_plan,
            generation,
            group_generation,
            topology,
            tp_bootstrap_source(),
            startups,
            inspections,
            device_plans,
            execution_plans,
            kv_plan,
            bootstrap_test_rendezvous(),
            collective_runtime_version_code=21403,
          ),
        )
      catch {
        Rejected(rule=WorkerContract, rank=1) => ()
        _ => fail("unexpected worker contract-domain rejection")
      } noraise {
        _ => fail("mixed worker sequence or query-limit domain was admitted")
      }
    }
    let valid_startups = [
      tp_bootstrap_startup(model_plan.identity(), 0),
      tp_bootstrap_startup(model_plan.identity(), 1),
    ]
    try
      ignore(
        admit(
          model_plan,
          generation,
          group_generation,
          topology,
          tp_bootstrap_source(),
          valid_startups,
          inspections,
          device_plans,
          [execution_plans[0]],
          kv_plan,
          bootstrap_test_rendezvous(),
          collective_runtime_version_code=21403,
        ),
      )
    catch {
      Rejected(rule=IncompleteRankSet, rank=-1) => ()
      _ => fail("unexpected incomplete execution-plan set rejection")
    } noraise {
      _ => fail("incomplete execution-plan set was admitted")
    }
    root.close()
  }
}
