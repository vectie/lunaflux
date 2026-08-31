///|
fn ml_limits(workspace? : Int64 = 4096L) -> AttentionMultiLaunchLimits raise {
  AttentionMultiLaunchLimits::new(
    maximum_workspace_bytes=workspace,
    maximum_symbol_bytes=256,
    maximum_operands_per_launch=16,
  )
}

///|
fn ml_target() -> @catalog.DeviceTarget raise {
  @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
}

///|
fn ml_profile() -> @launch_contract.PagedKernelProfile raise {
  @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=2,
    max_query_tokens=2,
    max_page_table_entries=16,
  )
}

///|
fn ml_family(digit : Char) -> @catalog.AotKernelFamily raise {
  @catalog.AotKernelFamily::new(
    module_digest=@catalog.AotArtifactDigest::from_sha256(
      digit.to_string().repeat(64),
    ),
    family_id=@catalog.AotKernelFamilyId::from_stable_int(7),
  )
}

///|
fn ml_entry(
  id : Int,
  digit? : Char = 'a',
) -> @catalog.AotKernelEntryPoint raise {
  @catalog.AotKernelEntryPoint::new(
    family=ml_family(digit),
    entry_point_id=@catalog.AotKernelEntryPointId::from_stable_int(id),
  )
}

///|
fn ml_operand(
  role : @launch_contract.AotOperandRole,
  bytes : Int64,
  alignment? : Int64 = 4L,
) -> @launch_contract.AotOperand raise {
  @launch_contract.AotOperand::new(role~, byte_count=bytes, alignment~)
}

///|
fn ml_workspace(
  limits? : AttentionMultiLaunchLimits = ml_limits(),
) -> AttentionDecodeWorkspaceContract raise {
  AttentionDecodeWorkspaceContract::new(
    scalar=F32,
    split_count=2,
    partial_count=8L,
    maximum_offset_bytes=0L,
    denominator_offset_bytes=32L,
    numerator_offset_bytes=64L,
    total_bytes=128L,
    alignment=16L,
    limits~,
  )
}

///|
fn ml_partial_operands(
  workspace : AttentionDecodeWorkspaceContract,
) -> Array[@launch_contract.AotOperand] raise {
  let layer = @plan.KvLayerId::new(0)
  [
    ml_operand(RuntimeInput(StepCounts), 8L),
    ml_operand(RuntimeInput(QueryPositions), 8L),
    ml_operand(RuntimeInput(QueryRowOffsets), 12L),
    ml_operand(RuntimeInput(SequenceLengths), 8L),
    ml_operand(RuntimeInput(PageTableOffsets), 12L),
    ml_operand(RuntimeInput(PageTablePageIndices), 64L),
    ml_operand(ActivationInput(@plan.ActivationRef::new(1)), 32L),
    ml_operand(PersistentKvComponent(layer, Key), 256L),
    ml_operand(PersistentKvComponent(layer, Value), 256L),
    ml_operand(
      Workspace,
      workspace.total_bytes(),
      alignment=workspace.alignment(),
    ),
  ]
}

///|
fn ml_merge_operands(
  workspace : AttentionDecodeWorkspaceContract,
) -> Array[@launch_contract.AotOperand] raise {
  [
    ml_operand(RuntimeInput(StepCounts), 8L),
    ml_operand(RuntimeInput(QueryRowOffsets), 12L),
    ml_operand(
      Workspace,
      workspace.total_bytes(),
      alignment=workspace.alignment(),
    ),
    ml_operand(ActivationOutput(@plan.ActivationRef::new(2)), 16L),
  ]
}

///|
fn ml_dimensions(z : Int) -> @launch_contract.AotLaunchDimensions raise {
  @launch_contract.AotLaunchDimensions::new(
    grid_x=2,
    grid_y=2,
    grid_z=z,
    block_x=32,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
}

///|
test "decode multi-launch admission owns exact ordered module and workspace" {
  let limits = ml_limits()
  let workspace = ml_workspace(limits~)
  let contract = admit_attention_decode_multilaunch_contract(
    operation_id=@plan.OperationId::new(9),
    profile=ml_profile(),
    target=ml_target(),
    workspace~,
    partial=AttentionOrderedLaunchInput::new(
      stage=Partial,
      entry_point=ml_entry(41),
      function_symbol="partial",
      dimensions=ml_dimensions(2),
      operands=ml_partial_operands(workspace),
    ),
    merge=AttentionOrderedLaunchInput::new(
      stage=Merge,
      entry_point=ml_entry(42),
      function_symbol="merge",
      dimensions=ml_dimensions(1),
      operands=ml_merge_operands(workspace),
    ),
    limits~,
  )
  assert_eq(contract.launches().length(), 2)
  assert_eq(contract.launches()[0].ordinal(), 0)
  assert_eq(contract.launches()[0].stage(), Partial)
  assert_eq(contract.launches()[1].ordinal(), 1)
  assert_eq(contract.launches()[1].stage(), Merge)
  assert_eq(contract.launches()[0].module_digest().sha256(), "a".repeat(64))
  assert_eq(contract.workspace().scalar(), F32)
  assert_eq(contract.workspace().total_bytes(), 128L)
}

///|
test "decode multi-launch rejects foreign module and reordered stages" {
  let limits = ml_limits()
  let workspace = ml_workspace(limits~)
  let partial = AttentionOrderedLaunchInput::new(
    stage=Partial,
    entry_point=ml_entry(41),
    function_symbol="partial",
    dimensions=ml_dimensions(2),
    operands=ml_partial_operands(workspace),
  )
  try
    admit_attention_decode_multilaunch_contract(
      operation_id=@plan.OperationId::new(9),
      profile=ml_profile(),
      target=ml_target(),
      workspace~,
      partial~,
      merge=AttentionOrderedLaunchInput::new(
        stage=Merge,
        entry_point=ml_entry(42, digit='b'),
        function_symbol="merge",
        dimensions=ml_dimensions(1),
        operands=ml_merge_operands(workspace),
      ),
      limits~,
    )
  catch {
    @luna_attention_multilaunch_contract.MismatchedAttentionMultiLaunchContract(
      Module
    ) => ()
    _ => fail("unexpected foreign-module rejection")
  } noraise {
    _ => fail("foreign module was admitted")
  }
  try
    admit_attention_decode_multilaunch_contract(
      operation_id=@plan.OperationId::new(9),
      profile=ml_profile(),
      target=ml_target(),
      workspace~,
      partial=AttentionOrderedLaunchInput::new(
        stage=Merge,
        entry_point=ml_entry(41),
        function_symbol="partial",
        dimensions=ml_dimensions(2),
        operands=ml_partial_operands(workspace),
      ),
      merge=AttentionOrderedLaunchInput::new(
        stage=Partial,
        entry_point=ml_entry(42),
        function_symbol="merge",
        dimensions=ml_dimensions(1),
        operands=ml_merge_operands(workspace),
      ),
      limits~,
    )
  catch {
    @luna_attention_multilaunch_contract.MismatchedAttentionMultiLaunchContract(
      Launch
    ) => ()
    _ => fail("unexpected launch-order rejection")
  } noraise {
    _ => fail("reordered stages were admitted")
  }
}

///|
test "decode workspace fails closed above its startup bound" {
  let limits = ml_limits(workspace=64L)
  try ml_workspace(limits~) catch {
    @luna_attention_multilaunch_contract.InvalidAttentionMultiLaunchContract(
      Workspace
    ) => ()
    _ => fail("unexpected workspace-limit rejection")
  } noraise {
    _ => fail("oversized workspace was admitted")
  }
}
