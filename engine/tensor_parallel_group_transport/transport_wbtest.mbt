///|
fn rebuild_group_inputs(
  base : TensorParallelGroupPlanningInputs,
  source : @worker_wire.EncodedBootstrapSource,
  declaration : @device_topology.LocalTopologyDeclaration,
  inspections : ArrayView[@device_materialize.DeviceShardedWeightFileInspection],
  executable_digest? : String = "9".repeat(64),
) -> TensorParallelGroupPlanningInputs raise TensorParallelGroupTransportError {
  TensorParallelGroupPlanningInputs::new(
    source~,
    model_plan=base.model_plan,
    model_generation=base.model_generation,
    topology_declaration=declaration,
    topology=base.topology,
    inspection_blueprints=inspections,
    device_step_limits=base.device_step_limits,
    manifest_limits=base.manifest_limits,
    launch_limits=base.launch_limits,
    execution_limits=base.execution_limits,
    worker_limits=base.worker_limits,
    inference_limits=base.inference_limits,
    process_limits=base.process_limits,
    admitted_rank_child_activation_path=base.admitted_rank_child_activation_path,
    rank_child_executable_digest_sha256=executable_digest,
  )
}

///|
async test "group template digest rejects zero and binds executable substitution" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-tp-template-hostile")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let fixture = group_fixture(directory)
    let base = fixture.inputs
    let alternate = rebuild_group_inputs(
      base,
      base.source,
      base.topology_declaration,
      base.inspection_blueprints,
      executable_digest="8".repeat(64),
    )
    assert_true(base.template_digest() != alternate.template_digest())
    expect_group_rule(
      () => {
        ignore(
          rebuild_group_inputs(
            base,
            base.source,
            base.topology_declaration,
            base.inspection_blueprints,
            executable_digest="0".repeat(64),
          ),
        )
      },
      Source,
    )
  }
}

///|
fn expect_group_rule(
  body : () -> Unit raise,
  expected : TensorParallelGroupTransportRule,
) -> Unit raise {
  try body() catch {
    Rejected(rule~, rank=_) => assert_eq(rule, expected)
    _ => fail("unexpected group-transport rejection")
  } noraise {
    _ => fail("hostile group-transport evidence was admitted")
  }
}

///|
async test "group owner retains one root pair and advances generation and predecessor" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-tp-group-owner")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let fixture = group_fixture(directory)
    let owner = prepare_tensor_parallel_group_transport(
      fixture.model_root,
      fixture.kernel_root,
      fixture.inputs,
      fixture.plan_identities,
      previous_plan_sequence_value=0UL,
    )
    fixture.model_root.close()
    fixture.kernel_root.close()
    assert_eq(owner.lifecycle(), Prepared)
    assert_eq(owner.world_size(), 2)
    assert_eq(owner.model_generation_value(), 7UL)
    assert_eq(owner.previous_plan_sequence_value(), 0UL)
    assert_false(owner.has_recovery_obligation())
    // Idle rank loss carries sequence zero, so replacement authority must be
    // retained independently of an active scheduler flight.
    owner.replacement_required = true
    assert_true(owner.has_recovery_obligation())
    owner.replacement_required = false
    assert_eq(owner.mint_group_generation().value(), 1UL)
    assert_eq(owner.mint_group_generation().value(), 2UL)
    owner.validate_service_identity(
      model_identity=fixture.inputs.model_identity(),
      model_generation_value=7UL,
      bootstrap_source_digest=fixture.inputs.bootstrap_source_digest(),
      worker_limits=fixture.inputs.worker_limits(),
      inference_limits=fixture.inputs.inference_limits(),
      previous_plan_sequence_value=0UL,
    )
    expect_group_rule(
      () => {
        owner.validate_service_identity(
          model_identity=fixture.inputs.model_identity(),
          model_generation_value=8UL,
          bootstrap_source_digest=fixture.inputs.bootstrap_source_digest(),
          worker_limits=fixture.inputs.worker_limits(),
          inference_limits=fixture.inputs.inference_limits(),
          previous_plan_sequence_value=0UL,
        )
      },
      ModelGeneration,
    )
    expect_group_rule(
      () => {
        owner.validate_service_identity(
          model_identity=fixture.inputs.model_identity(),
          model_generation_value=7UL,
          bootstrap_source_digest=fixture.inputs.bootstrap_source_digest(),
          worker_limits=fixture.inputs.worker_limits(),
          inference_limits=fixture.inputs.inference_limits(),
          previous_plan_sequence_value=1UL,
        )
      },
      SchedulerPredecessor,
    )
    owner.advance_scheduler_predecessor(1UL)
    assert_eq(owner.previous_plan_sequence_value(), 1UL)
    expect_group_rule(
      () => owner.advance_scheduler_predecessor(1UL),
      SchedulerPredecessor,
    )
    expect_group_rule(
      () => owner.advance_scheduler_predecessor(3UL),
      SchedulerPredecessor,
    )
    owner.advance_scheduler_predecessor(2UL)
    assert_eq(owner.previous_plan_sequence_value(), 2UL)
    owner.begin_close()
    assert_eq(owner.lifecycle(), Closed)
  }
}

///|
async test "planning rejects non-v5 rank topology sequence and owner substitution" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-tp-group-hostile")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let fixture = group_fixture(directory)
    let base = fixture.inputs
    let source_roots = base.source.roots()
    let ordinary = group_fixture_ordinary_source(
      source_roots.model_root(),
      source_roots.kernel_root(),
    )
    expect_group_rule(
      () => {
        ignore(
          rebuild_group_inputs(
            base,
            ordinary,
            base.topology_declaration,
            base.inspection_blueprints,
          ),
        )
      },
      Source,
    )
    expect_group_rule(
      () => {
        ignore(
          rebuild_group_inputs(base, base.source, base.topology_declaration, [
            base.inspection_blueprints[1],
            base.inspection_blueprints[0],
          ]),
        )
      },
      RankInspection,
    )
    let first = base.topology_declaration.rank(0).unwrap()
    let second = base.topology_declaration.rank(1).unwrap()
    let changed_declaration = @device_topology.LocalTopologyDeclaration::new(
      [
        @device_topology.LocalTopologyRank::new(
          ordinal=first.ordinal(),
          name="substituted-rank",
          total_memory_bytes=first.total_memory_bytes(),
          target=first.target(),
        ),
        second,
      ],
      [@device_topology.LocalTopologyLink::direct(lower_rank=0, upper_rank=1)],
    )
    expect_group_rule(
      () => {
        ignore(
          rebuild_group_inputs(
            base,
            base.source,
            changed_declaration,
            base.inspection_blueprints,
          ),
        )
      },
      Topology,
    )
    expect_group_rule(
      () => {
        ignore(
          prepare_tensor_parallel_group_transport(
            fixture.model_root,
            fixture.kernel_root,
            base,
            [fixture.plan_identities[0], fixture.plan_identities[0]],
            previous_plan_sequence_value=0UL,
          ),
        )
      },
      SchedulerPredecessor,
    )
    expect_group_rule(
      () => {
        ignore(
          prepare_tensor_parallel_group_transport(
            fixture.model_root,
            fixture.kernel_root,
            base,
            fixture.plan_identities,
            previous_plan_sequence_value=0xffffffffffffffffUL,
          ),
        )
      },
      SchedulerPredecessor,
    )
    let foreign_path = "\{directory}/foreign"
    @fs.mkdir(foreign_path)
    let foreign = @approved_fs.ApprovedRoot::open_absolute(
      @fs.realpath(foreign_path),
    )
    expect_group_rule(
      () => {
        ignore(
          prepare_tensor_parallel_group_transport(
            foreign,
            fixture.kernel_root,
            base,
            fixture.plan_identities,
            previous_plan_sequence_value=0UL,
          ),
        )
      },
      Root,
    )
    foreign.close()
    fixture.model_root.close()
    fixture.kernel_root.close()
  }
}

///|
async test "planning accepts only the canonical approved v8 tensor parallel pair" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-tp-group-approved-v8")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let fixture = group_fixture(directory)
    let approved = group_fixture_approved_source(fixture.inputs.source)
    let inputs = rebuild_group_inputs(
      fixture.inputs,
      approved,
      fixture.inputs.topology_declaration,
      fixture.inputs.inspection_blueprints,
    )
    assert_eq(
      inputs.source.recipe(),
      DenseLlamaTensorParallelPagedAotApprovedV8,
    )
    assert_true(inputs.source.has_valid_tensor_parallel_recipe_binding())
    fixture.model_root.close()
    fixture.kernel_root.close()
  }
}

///|
async test "malformed failure publication remains rejected on retry" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-tp-group-failure-hostile")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let fixture = group_fixture(directory)
    let owner = prepare_tensor_parallel_group_transport(
      fixture.model_root,
      fixture.kernel_root,
      fixture.inputs,
      fixture.plan_identities,
      previous_plan_sequence_value=0UL,
    )
    fixture.model_root.close()
    fixture.kernel_root.close()
    let protocol = @rank_group_protocol.RankGroupOwner::new(
      identity=fixture.inputs.model_identity(),
      model_plan_generation=@worker_protocol.ModelPlanGeneration::from_wire(7UL),
      generation=@rank_group_protocol.RankGroupGeneration::first(),
      world_size=2,
      previous_plan_sequence_value=0UL,
      plan_buffer_identities=fixture.plan_identities,
    )
    protocol.report_idle_rank_loss(0)
    let report = protocol.take_failure()
    for _ in 0..<2 {
      expect_group_rule(
        () => owner.validate_or_poison_group_failure_report(report, 2UL, 2),
        Process,
      )
      assert_true(owner.retained_failure_report is None)
      assert_false(owner.failure_report_taken)
      assert_true(owner.invalid_failure_report)
      assert_true(owner.has_recovery_obligation())
    }
    owner.begin_close()
  }
}

///|
async test "replacement foundations reject changed shard and manifest digest" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-tp-group-source-hostile")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let fixture = group_fixture(directory)
    let inputs = fixture.inputs
    let runtime = @tensor_parallel_worker_bootstrap.admit_runtime_policy(
      inputs.source,
      inputs.topology,
    )
    let (inspections, device_plans) = fresh_rank_foundations(
      inputs,
      fixture.model_root,
      runtime,
    )
    @fs.write_file(
      "\{directory}/kernel/execution.json",
      b"{}",
      create_mode=CreateNew,
    )
    expect_group_rule(
      () => {
        ignore(
          fresh_execution_inputs(
            inputs,
            0UL,
            fixture.kernel_root,
            runtime,
            inspections,
            device_plans,
          ),
        )
      },
      Manifest,
    )
    let changed = group_fixture_weight_file().to_array()
    changed[changed.length() - 1] = b'!'
    @fs.write_file(
      "\{directory}/model/weights.safetensors",
      Bytes::from_array(changed),
      create_mode=CreateOrTruncate,
    )
    expect_group_rule(
      () => ignore(fresh_rank_foundations(inputs, fixture.model_root, runtime)),
      RankInspection,
    )
    fixture.model_root.close()
    fixture.kernel_root.close()
  }
}
