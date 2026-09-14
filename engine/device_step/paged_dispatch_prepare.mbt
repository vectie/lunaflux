///|
/// The largest profile bucket is still a bucket. Artifact launch envelopes may
/// overprovision grid-stride kernels; they are not the live profile geometry.
fn paged_maximum_profile_query_bound(
  profile : @launch_contract.PagedKernelProfile,
  has_launch_contract : Bool,
  decode : Bool,
) -> Int? {
  if !has_launch_contract {
    None
  } else {
    Some(
      if decode {
        profile.max_query_rows()
      } else {
        profile.max_query_tokens()
      },
    )
  }
}

///|
fn prepare_paged_ordered_executors(
  context : @device.Context,
  steps : ReadOnlyArray[PreparedPagedGraphStep],
  resources : PagedGraphResources,
  graph_policy : PagedGraphExecutionPolicy,
  table : @luna_execution_graph_strategy.ExecutionGraphBucketTable,
  profile : @launch_contract.PagedKernelProfile,
  measured_attention_candidates? : ReadOnlyArray[Int]? = None,
  prefill_wide_steps? : ReadOnlyArray[PreparedPagedGraphStep]? = None,
  prefill_wide_minimum_query_tokens? : Int = PAGED_DECODE_SPLIT_DISABLED_CONTEXT_THRESHOLD,
  prefill_split_steps? : ReadOnlyArray[PreparedPagedGraphStep]? = None,
  prefill_split_minimum_context_tokens? : Int = PAGED_DECODE_SPLIT_DISABLED_CONTEXT_THRESHOLD,
  prefill_split_query_tile? : Int = 1,
  prefill_split_partition_count? : Int = 1,
  deep_prefill_split_steps? : ReadOnlyArray[PreparedPagedGraphStep]? = None,
  deep_prefill_split_minimum_context_tokens? : Int = PAGED_DECODE_SPLIT_DISABLED_CONTEXT_THRESHOLD,
  deep_prefill_split_query_tile? : Int = 1,
  deep_prefill_split_partition_count? : Int = 1,
  decode_steps? : ReadOnlyArray[PreparedPagedGraphStep]? = None,
  decode_split_steps? : ReadOnlyArray[PreparedPagedGraphStep]? = None,
  decode_split_minimum_context_tokens? : Int = PAGED_DECODE_SPLIT_DISABLED_CONTEXT_THRESHOLD,
  decode_split_maximum_batch_rows? : Int = 2147483647,
) -> PagedOrderedExecutorDispatch raise DeviceStepError {
  let maximum_query_bound = paged_maximum_profile_query_bound(
    profile,
    resources.graph_launch_contract is Some(_),
    false,
  )
  let maximum_decode_bound = paged_maximum_profile_query_bound(
    profile,
    resources.graph_launch_contract is Some(_),
    true,
  )
  let maximum_launches = paged_ordered_launches(
    steps,
    resources,
    query_token_bound=maximum_query_bound,
  )
  let baseline_operation_ids = ReadOnlyArray::from_array(
    steps.map(step => step.operation_id),
  )
  let slot_owner_indices = FixedArray::make(table.bucket_count(), -1)
  let decode_split_slot_owner_indices = FixedArray::make(
    table.bucket_count(),
    -1,
  )
  let prefill_wide_slot_owner_indices = FixedArray::make(
    table.bucket_count(),
    -1,
  )
  let prefill_split_slot_owner_indices = FixedArray::make(
    table.bucket_count(),
    -1,
  )
  let mut fallback_owner_index = -1
  let telemetry_owner_index = match graph_policy {
    PagedOrderedEagerOnly => {
      if resources.graph_launch_contract is Some(_) {
        for bound in paged_prefill_capture_bounds(profile) {
          let launches = paged_ordered_launches(
            steps,
            resources,
            query_token_bound=Some(bound),
          )
          let owner = prepare_one_paged_ordered_executor(
            context,
            launches[:],
            baseline_operation_ids,
            resources,
            OrderedKernelEagerOnly,
            source_steps=steps,
          )
          map_paged_prefill_capture_owner(
            table,
            profile.max_query_rows(),
            bound,
            owner,
            slot_owner_indices,
          )
        }
        if decode_steps is None {
          for bound in paged_decode_capture_bounds(profile) {
            let launches = paged_ordered_launches(
              steps,
              resources,
              query_token_bound=Some(bound),
            )
            let owner = prepare_one_paged_ordered_executor(
              context,
              launches[:],
              baseline_operation_ids,
              resources,
              OrderedKernelEagerOnly,
              source_steps=steps,
            )
            map_paged_decode_capture_owner(
              table, bound, bound, owner, slot_owner_indices,
            )
          }
        }
      }
      fallback_owner_index = prepare_one_paged_ordered_executor(
        context,
        maximum_launches[:],
        baseline_operation_ids,
        resources,
        OrderedKernelEagerOnly,
        source_steps=steps,
      )
      fallback_owner_index
    }
    PagedCapturedRequired | PagedCapturedWithEagerFallback => {
      // Qwen's composite launch contract can safely specialize fixed graph
      // geometry while retaining the same reusable descriptor and allocation
      // addresses. Capture the common decode buckets up front so c1/c8 do not
      // replay the maximum-token grid full of empty blocks. The maximum graph
      // remains the universal fallback for prefill, mixed, and uncommon decode
      // shapes.
      if resources.graph_launch_contract is Some(_) {
        for bound in paged_prefill_capture_bounds(profile) {
          let launches = paged_ordered_launches(
            steps,
            resources,
            query_token_bound=Some(bound),
          )
          let owner = prepare_one_paged_ordered_executor(
            context,
            launches[:],
            baseline_operation_ids,
            resources,
            paged_ordered_device_policy(graph_policy),
            source_steps=steps,
          )
          map_paged_prefill_capture_owner(
            table,
            profile.max_query_rows(),
            bound,
            owner,
            slot_owner_indices,
          )
        }
        if decode_steps is None {
          for bound in paged_decode_capture_bounds(profile) {
            let launches = paged_ordered_launches(
              steps,
              resources,
              query_token_bound=Some(bound),
            )
            let owner = prepare_one_paged_ordered_executor(
              context,
              launches[:],
              baseline_operation_ids,
              resources,
              paged_ordered_device_policy(graph_policy),
              source_steps=steps,
            )
            map_paged_decode_capture_owner(
              table, bound, bound, owner, slot_owner_indices,
            )
          }
        }
      }
      let maximum = prepare_one_paged_ordered_executor(
        context,
        maximum_launches[:],
        baseline_operation_ids,
        resources,
        paged_ordered_device_policy(graph_policy),
        source_steps=steps,
      )
      for slot in paged_maximum_graph_slots(table, profile) {
        if slot_owner_indices[slot] < 0 {
          slot_owner_indices[slot] = maximum
        }
      }
      // Capture-with-fallback reports OrderedKernelEager from this same owner
      // when the backend lacks graph support; no duplicate launch list is
      // needed. Non-maximum buckets therefore use the selected maximum owner
      // in either backend mode.
      fallback_owner_index = maximum
      maximum
    }
  }
  let prefill_split_owner_index = match prefill_split_steps {
    Some(selected) => {
      let operation_ids = ReadOnlyArray::from_array(
        selected.map(step => step.operation_id),
      )
      if resources.graph_launch_contract is Some(_) {
        for bound in paged_prefill_capture_bounds(profile) {
          let launches = paged_ordered_launches(
            selected,
            resources,
            query_token_bound=Some(bound),
            phase=SplitPrefillLaunch,
          )
          let owner = prepare_one_paged_ordered_executor(
            context,
            launches[:],
            operation_ids,
            resources,
            paged_ordered_device_policy(graph_policy),
            source_steps=selected,
          )
          map_paged_prefill_capture_owner(
            table, 1, bound, owner, prefill_split_slot_owner_indices,
          )
        }
      }
      let launches = paged_ordered_launches(
        selected,
        resources,
        query_token_bound=maximum_query_bound,
        phase=SplitPrefillLaunch,
      )
      prepare_one_paged_ordered_executor(
        context,
        launches[:],
        operation_ids,
        resources,
        paged_ordered_device_policy(graph_policy),
        source_steps=selected,
      )
    }
    None => -1
  }
  let deep_prefill_split_owner_index = match deep_prefill_split_steps {
    Some(selected) => {
      let launches = paged_ordered_launches(selected, resources)
      let operation_ids = ReadOnlyArray::from_array(
        selected.map(step => step.operation_id),
      )
      prepare_one_paged_ordered_executor(
        context,
        launches[:],
        operation_ids,
        resources,
        paged_ordered_device_policy(graph_policy),
        source_steps=selected,
      )
    }
    None => -1
  }
  let prefill_wide_owner_index = match prefill_wide_steps {
    Some(selected) => {
      let operation_ids = ReadOnlyArray::from_array(
        selected.map(step => step.operation_id),
      )
      if resources.graph_launch_contract is Some(_) {
        for bound in paged_prefill_capture_bounds(profile) {
          if bound >= prefill_wide_minimum_query_tokens {
            let launches = paged_ordered_launches(
              selected,
              resources,
              query_token_bound=Some(bound),
              phase=WidePrefillLaunch,
            )
            let owner = prepare_one_paged_ordered_executor(
              context,
              launches[:],
              operation_ids,
              resources,
              paged_ordered_device_policy(graph_policy),
              source_steps=selected,
            )
            map_paged_prefill_capture_owner(
              table,
              profile.max_query_rows(),
              bound,
              owner,
              prefill_wide_slot_owner_indices,
            )
          }
        }
      }
      let launches = paged_ordered_launches(
        selected,
        resources,
        query_token_bound=maximum_query_bound,
        phase=WidePrefillLaunch,
      )
      prepare_one_paged_ordered_executor(
        context,
        launches[:],
        operation_ids,
        resources,
        paged_ordered_device_policy(graph_policy),
        source_steps=selected,
      )
    }
    None => -1
  }
  let decode_owner_index = match decode_steps {
    Some(selected) => {
      let operation_ids = ReadOnlyArray::from_array(
        selected.map(step => step.operation_id),
      )
      if resources.graph_launch_contract is Some(_) {
        clear_paged_decode_capture_owners(
          table,
          profile.max_query_rows(),
          profile.max_query_rows(),
          slot_owner_indices,
        )
        for bound in paged_decode_capture_bounds(profile) {
          let launches = paged_ordered_launches(
            selected,
            resources,
            query_token_bound=Some(bound),
            phase=DecodeLaunch,
          )
          let owner = prepare_one_paged_ordered_executor(
            context,
            launches[:],
            operation_ids,
            resources,
            paged_ordered_device_policy(graph_policy),
            source_steps=selected,
          )
          map_paged_decode_capture_owner(
            table, bound, bound, owner, slot_owner_indices,
          )
        }
      }
      let maximum_launches = paged_ordered_launches(
        selected,
        resources,
        query_token_bound=maximum_decode_bound,
        phase=DecodeLaunch,
      )
      prepare_one_paged_ordered_executor(
        context,
        maximum_launches[:],
        operation_ids,
        resources,
        paged_ordered_device_policy(graph_policy),
        source_steps=selected,
      )
    }
    None => -1
  }
  let decode_split_owner_index = match decode_split_steps {
    Some(selected) => {
      let launches = paged_ordered_launches(selected, resources)
      let operation_ids = ReadOnlyArray::from_array(
        selected.map(step => step.operation_id),
      )
      prepare_decode_split_bucket_owners(
        context,
        table,
        profile,
        match decode_steps {
          Some(decode) => decode
          None => steps
        },
        selected,
        operation_ids,
        resources,
        graph_policy,
        decode_split_slot_owner_indices,
      )
      prepare_one_paged_ordered_executor(
        context,
        launches[:],
        operation_ids,
        resources,
        paged_ordered_device_policy(graph_policy),
        source_steps=selected,
      )
    }
    None => -1
  }
  {
    measured_owner_indices: map_measured_attention_owners(
      measured_attention_candidates, slot_owner_indices, fallback_owner_index, prefill_wide_slot_owner_indices,
      prefill_wide_owner_index,
    ),
    slot_owner_indices,
    prefill_wide_slot_owner_indices,
    prefill_split_slot_owner_indices,
    decode_split_slot_owner_indices,
    fallback_owner_index,
    telemetry_owner_index,
    prefill_wide_owner_index,
    prefill_wide_minimum_query_tokens,
    prefill_split_owner_index,
    prefill_split_minimum_context_tokens,
    prefill_split_query_tile,
    prefill_split_partition_count,
    deep_prefill_split_owner_index,
    deep_prefill_split_minimum_context_tokens,
    deep_prefill_split_query_tile,
    deep_prefill_split_partition_count,
    decode_owner_index,
    decode_split_owner_index,
    decode_split_minimum_context_tokens,
    decode_split_maximum_batch_rows,
  }
}
