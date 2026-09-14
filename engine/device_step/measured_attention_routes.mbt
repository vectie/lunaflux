///|
/// Candidate 1 is the baseline complete graph; 2 is its wide-prefill variant.
/// The immutable measurements are consumed once, before resource preparation.
fn admit_qwen_attention_routes(
  runtime : @luna_fused_artifact_admission.ApprovedReusableFusedRuntimeBundle,
  profile : @launch_contract.PagedKernelProfile,
  has_wide : Bool,
) -> ReadOnlyArray[Int]? raise FusedGroupedOperationError {
  guard runtime.attention_bucket_routes() is Some(bytes) else { return None }
  if !has_wide {
    raise MismatchedFusedGroupedOperation(KernelAbi)
  }
  let table = new_paged_graph_bucket_table(profile)
  let routes = @luna_attention_tuning.parse_attention_bucket_routes(
    bytes,
    scope=runtime.attention_route_scope() catch {
      _ => raise MismatchedFusedGroupedOperation(KernelAbi)
    },
    table,
    baseline=1,
    [1, 2],
  ) catch {
    _ => raise MismatchedFusedGroupedOperation(Launch)
  }
  let candidates = routes.bucket_candidates()
  for slot, candidate in candidates {
    let bucket = table.bucket_at(slot) catch {
      _ => raise MismatchedFusedGroupedOperation(Launch)
    }
    if candidate >= 0 && bucket.phase() == DecodeGraph {
      raise MismatchedFusedGroupedOperation(Launch)
    }
  }
  Some(candidates)
}

///|
fn qwen_measured_attention_candidates(
  spans : ReadOnlyArray[ApprovedQwenFusedAttentionSpan],
) -> ReadOnlyArray[Int]? {
  match spans {
    [first, ..] => first.measured_attention_candidates
    [] => None
  }
}

///|
/// Pure startup mapping. Missing measurements retain legacy policy; measured
/// baseline choices override heuristics just as measured variant choices do.
fn map_measured_attention_owners(
  candidates : ReadOnlyArray[Int]?,
  baseline : FixedArray[Int],
  baseline_fallback : Int,
  wide : FixedArray[Int],
  wide_fallback : Int,
) -> ReadOnlyArray[Int] raise DeviceStepError {
  guard candidates is Some(ids) else { return ReadOnlyArray::from_array([]) }
  if ids.length() != baseline.length() || ids.length() != wide.length() {
    raise InvalidExecutor(GraphBucket)
  }
  let owners = Array::makei(ids.length(), slot => {
    match ids[slot] {
      -1 => -1
      1 => if baseline[slot] >= 0 { baseline[slot] } else { baseline_fallback }
      2 => if wide[slot] >= 0 { wide[slot] } else { wide_fallback }
      _ => -1
    }
  })
  for slot, candidate in ids {
    if candidate != -1 && (candidate < 1 || candidate > 2 || owners[slot] < 0) {
      raise InvalidExecutor(GraphBucket)
    }
  }
  ReadOnlyArray::from_array(owners)
}
