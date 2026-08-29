///|
/// Canonical stochastic candidate order: greater logit first and lower token
/// ID first on ties. Whole-row finite validation precedes every caller of this
/// comparator, so the relation is a total order over distinct token IDs.
#inline(never)
fn preferred_before(
  logits : ArrayView[Double],
  left_token : Int,
  right_token : Int,
) -> Bool {
  let left = logits[left_token]
  let right = logits[right_token]
  left > right || (left == right && left_token < right_token)
}

///|
#inline(never)
fn worse_than(
  logits : ArrayView[Double],
  left_token : Int,
  right_token : Int,
) -> Bool {
  preferred_before(logits, right_token, left_token)
}

///|
/// Repairs a heap whose root is the canonically worst retained candidate.
#inline(never)
fn sift_worst_down(
  token_ids : FixedArray[Int],
  logits : ArrayView[Double],
  start : Int,
  end : Int,
) -> Unit {
  let mut root = start
  while root * 2 + 1 <= end {
    let child = root * 2 + 1
    let mut selected = child
    if child + 1 <= end &&
      worse_than(logits, token_ids[child + 1], token_ids[child]) {
      selected = child + 1
    }
    if !worse_than(logits, token_ids[selected], token_ids[root]) {
      break
    }
    token_ids.swap(root, selected)
    root = selected
  }
}

///|
#inline(never)
fn make_worst_heap(
  token_ids : FixedArray[Int],
  logits : ArrayView[Double],
  count : Int,
) -> Unit {
  if count <= 1 {
    return
  }
  let mut start = (count - 2) / 2
  for ;; {
    sift_worst_down(token_ids, logits, start, count - 1)
    if start == 0 {
      break
    }
    start -= 1
  }
}

///|
/// Converts an existing worst-root heap into canonical best-first order.
#inline(never)
fn drain_worst_heap(
  token_ids : FixedArray[Int],
  logits : ArrayView[Double],
  count : Int,
) -> Unit {
  let mut end = count - 1
  while end > 0 {
    token_ids.swap(end, 0)
    end -= 1
    sift_worst_down(token_ids, logits, 0, end)
  }
}

///|
/// In-place heap sort with the canonical total order. It uses no temporary
/// collection or recursive call stack.
#inline(never)
fn sort_candidates(
  token_ids : FixedArray[Int],
  logits : ArrayView[Double],
  count : Int,
) -> Unit {
  make_worst_heap(token_ids, logits, count)
  drain_worst_heap(token_ids, logits, count)
}

///|
/// Retains the exact best `count` tokens in O(V log count) worst-case work.
/// The first `count` scratch entries finish in the same order as a full sort.
#inline(never)
fn select_top_k_candidates(
  token_ids : FixedArray[Int],
  logits : ArrayView[Double],
  token_count : Int,
  count : Int,
  scratch : SamplerScratch,
) -> Unit {
  for token_id in 0..<count {
    token_ids[token_id] = token_id
    scratch.order_candidate_visits += 1
  }
  make_worst_heap(token_ids, logits, count)
  scratch.order_heap_width = count
  for token_id in count..<token_count {
    scratch.order_candidate_visits += 1
    if preferred_before(logits, token_id, token_ids[0]) {
      token_ids[0] = token_id
      scratch.order_replacement_sifts += 1
      sift_worst_down(token_ids, logits, 0, count - 1)
    }
  }
  drain_worst_heap(token_ids, logits, count)
  scratch.order_sorted_width = count
}

///|
#inline(never)
fn prepare_candidate_order(
  logits : ArrayView[Double],
  token_count : Int,
  top_k_count : Int,
  has_restrictive_top_k~ : Bool,
  scratch : SamplerScratch,
) -> Unit {
  scratch.order_candidate_visits = 0
  scratch.order_heap_width = 0
  scratch.order_sorted_width = 0
  scratch.order_replacement_sifts = 0
  if has_restrictive_top_k {
    select_top_k_candidates(
      scratch.token_ids,
      logits,
      token_count,
      top_k_count,
      scratch,
    )
  } else {
    for token_id in 0..<token_count {
      scratch.token_ids[token_id] = token_id
      scratch.order_candidate_visits += 1
    }
    scratch.order_heap_width = token_count
    sort_candidates(scratch.token_ids, logits, token_count)
    scratch.order_sorted_width = token_count
  }
  // These private scalars are observed only by white-box complexity tests.
  // Explicit reads keep strict non-test builds honest without a branch, raise,
  // allocation, or public diagnostic surface.
  ignore(scratch.order_candidate_visits)
  ignore(scratch.order_heap_width)
  ignore(scratch.order_sorted_width)
  ignore(scratch.order_replacement_sifts)
}
