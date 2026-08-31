///|
test "paged attention lowering is deterministic and preserves exact v3 ABI" {
  let first = pa_lowering()
  let second = pa_lowering()
  assert_eq(first, second)
  assert_eq(first.query_heads(), 2)
  assert_eq(first.kv_heads(), 1)
  assert_eq(first.head_dimension(), 2)
  assert_eq(first.tokens_per_page(), 2)
  assert_eq(first.page_stride_bytes(), 256L)
  assert_eq(first.physical_page_capacity(), 4L)
  assert_eq(first.launch_dimensions().grid_x(), 3)
  assert_eq(first.launch_dimensions().grid_y(), 2)
  assert_eq(first.launch_dimensions().block_x(), 32)
  assert_eq(first.launch_dimensions().shared_memory_bytes(), 0)
  assert_eq(first.operands().length(), 10)
  assert_eq(first.entry_point_id(), pa_entry(3).entry_point_id())
  assert_eq(
    first.source_digest().as_hex(),
    "93883127bd5039aaf8810b4eeaef89e7f7bc13e46df29b5ca0613cb7d0bb36e0",
  )
  assert_eq(
    first.recipe_digest().as_hex(),
    "b373b71fb2b38b2a935d95d853075d988de377559cb9aa4ba2d5c044e6bc7d20",
  )
  assert_eq(first.operands()[0].role(), RuntimeInput(StepCounts))
  assert_eq(
    first.operands()[6].role(),
    ActivationInput(@plan.ActivationRef::new(1)),
  )
  assert_eq(
    first.operands()[9].role(),
    PersistentKvComponent(@plan.KvLayerId::new(0), Value),
  )
}

///|
test "generated source covers mixed prefill decode without grid synchronization" {
  let lowering = pa_lowering()
  let source = @utf8.decode(lowering.source_bytes())
  assert_true(source.contains("const int prefill_count = counts[0]"))
  assert_true(source.contains("decode_count = counts[1]"))
  assert_true(source.contains("key_cache[cache_index] = qkv["))
  assert_true(source.contains("in_chunk") == false)
  assert_true(source.contains("key_position >= chunk_start"))
  assert_true(source.contains("maximum = -CUDART_INF_F"))
  assert_true(source.contains("accumulator[item] / denominator"))
  assert_true(source.contains("__shfl_down_sync"))
  assert_false(source.contains("__syncthreads"))
  assert_false(source.contains("malloc"))
  let recipe = @utf8.decode(lowering.recipe_bytes())
  assert_true(recipe.contains("mode=mixed-prefill-decode\n"))
  assert_true(recipe.contains("language_standard=c++17\n"))
  assert_true(recipe.contains("grid=3,2,1\n"))
  assert_true(recipe.contains("block=32,1,1\n"))
  assert_true(recipe.contains("shared_memory_bytes=0\n"))
  assert_true(recipe.contains("manifest_bindable=false\n"))
  assert_false(recipe.contains("module_sha256="))
  assert_true(recipe.contains("kv_write=fused-current-token\n"))
  assert_true(recipe.contains("in_chunk_kv_source=qkv-activation\n"))
  assert_true(recipe.contains("fmad=false\n"))
}

///|
test "phase bundle emits independent prefill and decode entry points" {
  let contract = pa_attention_contract()
  let bundle = lower_paged_attention_cuda_aot_phase_bundle(
    pa_plan().operations()[2],
    contract.profile(),
    pa_layout(),
    @catalog.AotKernelEntryPointId::from_stable_int(31),
    @catalog.AotKernelEntryPointId::from_stable_int(32),
    pa_target(),
    pa_compiler(),
    contract.operands(),
    pa_limits(),
  )
  assert_eq(bundle.prefill().execution_phase(), Prefill)
  assert_eq(bundle.decode().execution_phase(), Decode)
  assert_true(
    @utf8.decode(bundle.prefill().source_bytes()).contains(
      "if (row >= prefill_count) continue;",
    ),
  )
  assert_true(
    @utf8.decode(bundle.decode().source_bytes()).contains(
      "row < prefill_count || row_end - row_start != 1",
    ),
  )
  let prefill_recipe = @utf8.decode(bundle.prefill().recipe_bytes())
  let decode_recipe = @utf8.decode(bundle.decode().recipe_bytes())
  assert_true(prefill_recipe.contains("mode=prefill\n"))
  assert_true(prefill_recipe.contains("attention_traversal=query-tile\n"))
  assert_true(decode_recipe.contains("mode=decode\n"))
  assert_true(decode_recipe.contains("attention_traversal=key-value-split\n"))
}

///|
test "binder rejects foreign geometry and lowering rejects target/compiler semantics" {
  let candidate = pa_lowering()
  try
    @luna_cuda_paged_attention_aot.bind_manifest_paged_attention_cuda_aot(
      candidate,
      pa_plan().operations()[2],
      pa_contracts(exact_dimensions=false),
    )
  catch {
    @luna_cuda_paged_attention_aot.Mismatch(Contract) => ()
    _ => fail("unexpected dimension rejection")
  } noraise {
    _ => fail("expected exact dimension rejection")
  }
  try {
    let contract = pa_attention_contract()
    @luna_cuda_paged_attention_aot.lower_paged_attention_cuda_aot_candidate(
      pa_plan().operations()[2],
      contract.profile(),
      pa_layout(),
      contract.entry_point().entry_point_id(),
      pa_target(bf16=false),
      pa_compiler(),
      contract.operands(),
      pa_limits(),
    )
  } catch {
    @luna_cuda_paged_attention_aot.Unsupported(DeviceTarget) => ()
    _ => fail("unexpected target rejection")
  } noraise {
    _ => fail("expected BF16 target rejection")
  }
  try {
    let contract = pa_attention_contract()
    @luna_cuda_paged_attention_aot.lower_paged_attention_cuda_aot_candidate(
      pa_plan().operations()[2],
      contract.profile(),
      pa_layout(),
      contract.entry_point().entry_point_id(),
      pa_target(),
      pa_compiler(reassociate=true),
      contract.operands(),
      pa_limits(),
    )
  } catch {
    @luna_cuda_paged_attention_aot.Unsupported(CompilerPolicy) => ()
    _ => fail("unexpected compiler rejection")
  } noraise {
    _ => fail("expected reassociation rejection")
  }
}

///|
test "strict binder authenticates final attention module after compilation" {
  let candidate = pa_lowering()
  let contracts = pa_contracts()
  let bound = @luna_cuda_paged_attention_aot.bind_manifest_paged_attention_cuda_aot(
    candidate,
    pa_plan().operations()[2],
    contracts,
  )
  assert_true(bound.manifest_bindable())
  assert_eq(bound.candidate(), candidate)
  assert_eq(bound.entry_point(), pa_entry(3))
  let record = @utf8.decode(bound.binding_record_bytes())
  assert_true(record.contains("module_sha256=" + "3".repeat(64)))
  assert_true(record.contains("manifest_bindable=true\n"))
}

///|
test "digest constructors reject non-canonical values" {
  try
    @luna_cuda_paged_attention_aot.PagedAttentionCudaSourceDigest::from_sha256(
      "ABC",
    )
  catch {
    Invalid(Source) => ()
    _ => fail("unexpected source digest rejection")
  } noraise {
    _ => fail("expected source digest rejection")
  }
}

///|
fn pa_decode_split_profile(
  max_page_table_entries : Int,
) -> @launch_contract.PagedKernelProfile raise {
  @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(9),
    max_query_rows=2,
    max_query_tokens=3,
    max_page_table_entries~,
  )
}

///|
fn pa_decode_split_operands(
  max_page_table_entries : Int,
) -> Array[@launch_contract.AotOperand] raise {
  let layer = @plan.KvLayerId::new(0)
  [
    pa_operand(RuntimeInput(StepCounts), 20L, 4L),
    pa_operand(RuntimeInput(QueryPositions), 12L, 4L),
    pa_operand(RuntimeInput(QueryRowOffsets), 12L, 4L),
    pa_operand(RuntimeInput(SequenceLengths), 8L, 4L),
    pa_operand(RuntimeInput(PageTableOffsets), 12L, 4L),
    pa_operand(
      RuntimeInput(PageTablePageIndices),
      max_page_table_entries.to_int64() * 4L,
      4L,
    ),
    pa_operand(ActivationInput(@plan.ActivationRef::new(1)), 48L, 16L),
    pa_operand(ActivationOutput(@plan.ActivationRef::new(2)), 24L, 16L),
    pa_operand(PersistentKvComponent(layer, Key), 1024L, 256L),
    pa_operand(PersistentKvComponent(layer, Value), 1024L, 256L),
  ]
}

///|
fn pa_decode_split_lowering(
  max_page_table_entries? : Int = 3,
  maximum_key_value_splits? : Int = 8,
  maximum_workspace_bytes? : Int64 = 1048576L,
) -> @luna_cuda_paged_attention_aot.PagedAttentionDecodeSplitCudaBundle raise {
  let profile = pa_decode_split_profile(max_page_table_entries)
  @luna_cuda_paged_attention_aot.lower_paged_attention_decode_split_cuda_aot_candidate(
    pa_plan().operations()[2],
    profile,
    pa_layout(),
    @catalog.AotKernelEntryPointId::from_stable_int(41),
    @catalog.AotKernelEntryPointId::from_stable_int(42),
    pa_target(),
    pa_compiler(),
    pa_decode_split_operands(max_page_table_entries),
    maximum_key_value_splits,
    maximum_workspace_bytes,
    pa_limits(),
  )
}

///|
test "decode split candidate is deterministic and one split covers the ordinary interval" {
  let first = pa_decode_split_lowering()
  let second = pa_decode_split_lowering()
  assert_eq(first, second)
  assert_false(first.manifest_bindable())
  assert_eq(first.workspace_plan().split_count(), 1)
  assert_eq(first.workspace_plan().partitions()[0].token_range(), (0, 6))
  assert_eq(first.workspace_plan().total_bytes(), 64L)
  let (partial, merge) = first.launch_dimensions()
  assert_eq((partial.grid_x(), partial.grid_y(), partial.grid_z()), (2, 2, 1))
  assert_eq((merge.grid_x(), merge.grid_y(), merge.grid_z()), (2, 2, 1))
  let source = @utf8.decode(first.source_bytes())
  assert_true(source.contains("decode_split_partial_v1_ep_41"))
  assert_true(source.contains("decode_split_merge_v1_ep_42"))
  assert_true(source.contains("for (int split = 0; split < 1; ++split)"))
  assert_true(source.contains("accumulator[item] / denominator"))
  let recipe = @utf8.decode(first.recipe_bytes())
  assert_true(recipe.contains("manifest_bindable=false\n"))
  assert_true(recipe.contains("partitions=0:0-6\n"))
  assert_true(recipe.contains("workspace_total_bytes=64\n"))
}

///|
test "decode split candidate uses bounded partial workspace and ascending merge" {
  let lowering = pa_decode_split_lowering(
    max_page_table_entries=1024,
    maximum_key_value_splits=8,
  )
  let workspace = lowering.workspace_plan()
  assert_eq(workspace.split_count(), 2)
  assert_eq(workspace.partitions()[0].token_range(), (0, 1024))
  assert_eq(workspace.partitions()[1].token_range(), (1024, 2048))
  assert_eq(workspace.total_bytes(), 128L)
  let (partial, merge) = lowering.launch_dimensions()
  assert_eq(partial.grid_z(), 2)
  assert_eq(merge.grid_z(), 1)
  let source = @utf8.decode(lowering.source_bytes())
  assert_true(source.contains("const int partition_begin_page = split * 512"))
  assert_true(source.contains("for (int split = 0; split < 2; ++split)"))
  assert_true(source.contains("part_denominator * new_scale"))
  assert_true(source.contains("partial_numerator[partial * 2ULL + d]"))
  assert_true(source.contains("!isfinite(part_denominator)"))
  assert_true(source.contains("current_physical_page"))
  let recipe = @utf8.decode(lowering.recipe_bytes())
  assert_true(recipe.contains("merge_order=ascending-split-index\n"))
  assert_true(recipe.contains("partitions=0:0-1024,1:1024-2048\n"))
}

///|
test "decode split source preserves uneven page-balanced partitions" {
  let lowering = pa_decode_split_lowering(
    max_page_table_entries=2049,
    maximum_key_value_splits=8,
  )
  let workspace = lowering.workspace_plan()
  assert_eq(workspace.split_count(), 4)
  assert_eq(workspace.partitions()[0].token_range(), (0, 1026))
  assert_eq(workspace.partitions()[1].token_range(), (1026, 2050))
  assert_eq(workspace.partitions()[2].token_range(), (2050, 3074))
  assert_eq(workspace.partitions()[3].token_range(), (3074, 4098))
  let source = @utf8.decode(lowering.source_bytes())
  assert_true(source.contains("const int partition_begin_page = split * 512"))
  assert_true(source.contains("+ (split < 1 ? split : 1);"))
  assert_true(
    source.contains("const int partition_pages = 512 + (split < 1 ? 1 : 0);"),
  )
  let recipe = @utf8.decode(lowering.recipe_bytes())
  assert_true(
    recipe.contains("partitions=0:0-1026,1:1026-2050,2:2050-3074,3:3074-4098\n"),
  )
}

///|
test "decode split candidate fails closed on workspace and entry geometry" {
  try pa_decode_split_lowering(maximum_workspace_bytes=16L) catch {
    @luna_cuda_paged_attention_aot.Unsupported(Profile) => ()
    _ => fail("unexpected workspace rejection")
  } noraise {
    _ => fail("expected workspace rejection")
  }
  let profile = pa_decode_split_profile(3)
  try
    @luna_cuda_paged_attention_aot.lower_paged_attention_decode_split_cuda_aot_candidate(
      pa_plan().operations()[2],
      profile,
      pa_layout(),
      @catalog.AotKernelEntryPointId::from_stable_int(41),
      @catalog.AotKernelEntryPointId::from_stable_int(41),
      pa_target(),
      pa_compiler(),
      pa_decode_split_operands(3),
      8,
      1048576L,
      pa_limits(),
    )
  catch {
    @luna_cuda_paged_attention_aot.Invalid(Operation) => ()
    _ => fail("unexpected entry rejection")
  } noraise {
    _ => fail("expected entry rejection")
  }
}

///|
fn pa_decode_split_multilaunch_contract(
  candidate : @luna_cuda_paged_attention_aot.PagedAttentionDecodeSplitCudaBundle,
  partial_symbol? : String = "",
  partial_entry_id? : Int = 41,
  workspace_partial_count? : Int64 = 0L,
) -> @luna_attention_multilaunch_contract.AttentionDecodeMultiLaunchContract raise {
  let limits = @luna_attention_multilaunch_contract.AttentionMultiLaunchLimits::new(
    maximum_workspace_bytes=1048576L,
    maximum_symbol_bytes=256,
    maximum_operands_per_launch=16,
  )
  let candidate_workspace = candidate.workspace_plan()
  let candidate_partial_count = candidate_workspace.partial_count()
  let partial_count = if workspace_partial_count == 0L {
    candidate_partial_count
  } else {
    workspace_partial_count
  }
  let scalar_bytes = partial_count * 4L
  let total_bytes = if partial_count == candidate_partial_count {
    candidate_workspace.total_bytes()
  } else {
    scalar_bytes * 2L +
    partial_count * candidate.head_dimension().to_int64() * 4L
  }
  let workspace = @luna_attention_multilaunch_contract.AttentionDecodeWorkspaceContract::new(
    scalar=F32,
    split_count=candidate_workspace.split_count(),
    partial_count~,
    maximum_offset_bytes=0L,
    denominator_offset_bytes=scalar_bytes,
    numerator_offset_bytes=scalar_bytes * 2L,
    total_bytes~,
    alignment=16L,
    limits~,
  )
  let source_operands = candidate.operands()
  let workspace_operand = pa_operand(
    Workspace,
    workspace.total_bytes(),
    workspace.alignment(),
  )
  let partial_operands = [
    source_operands[0],
    source_operands[1],
    source_operands[2],
    source_operands[3],
    source_operands[4],
    source_operands[5],
    source_operands[6],
    source_operands[8],
    source_operands[9],
    workspace_operand,
  ]
  let merge_operands = [
    source_operands[0],
    source_operands[2],
    workspace_operand,
    source_operands[7],
  ]
  let family = @catalog.AotKernelFamily::new(
    module_digest=@catalog.AotArtifactDigest::from_sha256("a".repeat(64)),
    family_id=@catalog.AotKernelFamilyId::from_stable_int(91),
  )
  let (candidate_partial_symbol, merge_symbol) = candidate.function_symbols()
  let (partial_dimensions, merge_dimensions) = candidate.launch_dimensions()
  @luna_attention_multilaunch_contract.admit_attention_decode_multilaunch_contract(
    operation_id=candidate.operation_id(),
    profile=candidate.profile(),
    target=candidate.target(),
    workspace~,
    partial=@luna_attention_multilaunch_contract.AttentionOrderedLaunchInput::new(
      stage=Partial,
      entry_point=@catalog.AotKernelEntryPoint::new(
        family~,
        entry_point_id=@catalog.AotKernelEntryPointId::from_stable_int(
          partial_entry_id,
        ),
      ),
      function_symbol=if partial_symbol.is_empty() {
        candidate_partial_symbol
      } else {
        partial_symbol
      },
      dimensions=partial_dimensions,
      operands=partial_operands,
    ),
    merge=@luna_attention_multilaunch_contract.AttentionOrderedLaunchInput::new(
      stage=Merge,
      entry_point=@catalog.AotKernelEntryPoint::new(
        family~,
        entry_point_id=@catalog.AotKernelEntryPointId::from_stable_int(42),
      ),
      function_symbol=merge_symbol,
      dimensions=merge_dimensions,
      operands=merge_operands,
    ),
    limits~,
  )
}

///|
test "decode split production binding authenticates two ordered launches" {
  let candidate = pa_decode_split_lowering(max_page_table_entries=1024)
  let contract = pa_decode_split_multilaunch_contract(candidate)
  let binding = @luna_cuda_paged_attention_aot.bind_manifest_paged_attention_decode_split_cuda_aot(
    candidate,
    pa_plan().operations()[2],
    contract,
  )
  assert_true(binding.manifest_bindable())
  assert_eq(binding.candidate(), candidate)
  assert_eq(binding.contract(), contract)
  let launches = binding.contract().launches()
  assert_eq(launches.length(), 2)
  assert_eq(launches[0].stage(), Partial)
  assert_eq(launches[1].stage(), Merge)
  assert_eq(launches[0].module_digest().sha256(), "a".repeat(64))
  let record = @utf8.decode(binding.binding_record_bytes())
  assert_true(record.contains("launch_count=2\n"))
  assert_true(record.contains("launch_order=partial,merge\n"))
  assert_true(record.contains("workspace_scalar=f32\n"))
  assert_true(record.contains("workspace=128/16\n"))
  assert_true(record.contains("manifest_bindable=true\n"))
}

///|
test "decode split production binding rejects symbol entry and workspace drift" {
  let candidate = pa_decode_split_lowering(max_page_table_entries=1024)
  try
    @luna_cuda_paged_attention_aot.bind_manifest_paged_attention_decode_split_cuda_aot(
      candidate,
      pa_plan().operations()[2],
      pa_decode_split_multilaunch_contract(
        candidate,
        partial_symbol="foreign_partial",
      ),
    )
  catch {
    @luna_cuda_paged_attention_aot.Mismatch(Contract) => ()
    _ => fail("unexpected symbol rejection")
  } noraise {
    _ => fail("foreign symbol was bound")
  }
  try
    @luna_cuda_paged_attention_aot.bind_manifest_paged_attention_decode_split_cuda_aot(
      candidate,
      pa_plan().operations()[2],
      pa_decode_split_multilaunch_contract(candidate, partial_entry_id=43),
    )
  catch {
    @luna_cuda_paged_attention_aot.Mismatch(Contract) => ()
    _ => fail("unexpected entry rejection")
  } noraise {
    _ => fail("foreign entry point was bound")
  }
  try
    @luna_cuda_paged_attention_aot.bind_manifest_paged_attention_decode_split_cuda_aot(
      candidate,
      pa_plan().operations()[2],
      pa_decode_split_multilaunch_contract(
        candidate,
        workspace_partial_count=16L,
      ),
    )
  catch {
    @luna_cuda_paged_attention_aot.Mismatch(Contract) => ()
    _ => fail("unexpected workspace rejection")
  } noraise {
    _ => fail("foreign workspace was bound")
  }
}
