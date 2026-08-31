///|
fn broad_serving_wb_pair_sample(
  accepts? : UInt64 = 1UL,
  disconnects? : UInt64 = 0UL,
  backpressure? : UInt64 = 1UL,
) -> BroadServingMetricSample {
  {
    admissions: 3UL,
    completions: 2UL,
    cancellations: 1UL,
    deadlines: 0UL,
    failures: 0UL,
    prompt_tokens: 3UL,
    generated_tokens: 3UL,
    worker_failures: 0UL,
    backpressure,
    network_accepts: accepts,
    network_disconnects: disconnects,
    network_rejections: 0UL,
    queue_depth: 0,
    active_requests: 0,
    kv_pages_used: 0,
    kv_pages_free: 32,
  }
}

///|
fn broad_serving_wb_cancel_boundary_sample(
  admissions : UInt64,
  queue_depth? : Int = 2,
  active_requests? : Int = 0,
  generated_tokens? : UInt64 = 0UL,
) -> BroadServingMetricSample {
  {
    admissions,
    completions: 0UL,
    cancellations: 0UL,
    deadlines: 0UL,
    failures: 0UL,
    prompt_tokens: 0UL,
    generated_tokens,
    worker_failures: 0UL,
    backpressure: 0UL,
    network_accepts: 2UL,
    network_disconnects: 0UL,
    network_rejections: 0UL,
    queue_depth,
    active_requests,
    kv_pages_used: 0,
    kv_pages_free: 32,
  }
}

///|
test "broad serving pair metrics distinguish server and framed pool ownership" {
  assert_true(
    broad_serving_pair_metrics_valid(broad_serving_wb_pair_sample(), 1, 32),
  )
  assert_false(
    broad_serving_pair_metrics_valid(
      broad_serving_wb_pair_sample(),
      1,
      32,
      pooled=true,
    ),
  )
  let pooled = broad_serving_wb_pair_sample(
    accepts=3UL,
    disconnects=3UL,
    backpressure=0UL,
  )
  assert_true(broad_serving_pair_metrics_valid(pooled, 1, 32, pooled=true))
  assert_false(broad_serving_pair_metrics_valid(pooled, 1, 32))
}

///|
test "broad serving pool cancellation boundary preserves queued concurrency" {
  assert_true(
    broad_serving_pool_cancel_boundary_valid(
      broad_serving_wb_cancel_boundary_sample(1UL),
    ),
  )
  assert_true(
    broad_serving_pool_cancel_boundary_valid(
      broad_serving_wb_cancel_boundary_sample(
        2UL,
        queue_depth=1,
        active_requests=1,
      ),
    ),
  )
  assert_false(
    broad_serving_pool_cancel_boundary_valid(
      broad_serving_wb_cancel_boundary_sample(0UL),
    ),
  )
  assert_false(
    broad_serving_pool_cancel_boundary_valid(
      broad_serving_wb_cancel_boundary_sample(3UL),
    ),
  )
  assert_false(
    broad_serving_pool_cancel_boundary_valid(
      broad_serving_wb_cancel_boundary_sample(1UL, queue_depth=1),
    ),
  )
  assert_false(
    broad_serving_pool_cancel_boundary_valid(
      broad_serving_wb_cancel_boundary_sample(1UL, generated_tokens=1UL),
    ),
  )
}

///|
test "broad serving pool cancellation progress admits only bounded transition" {
  let waiting = broad_serving_wb_cancel_boundary_sample(1UL)
  assert_true(broad_serving_pool_cancel_progress_valid(waiting))
  let admitted = broad_serving_wb_cancel_boundary_sample(
    2UL,
    queue_depth=1,
    active_requests=1,
  )
  assert_true(broad_serving_pool_cancel_progress_valid(admitted))
  assert_false(
    broad_serving_pool_cancel_progress_valid(
      broad_serving_wb_cancel_boundary_sample(0UL),
    ),
  )
  assert_false(
    broad_serving_pool_cancel_progress_valid(
      broad_serving_wb_cancel_boundary_sample(3UL),
    ),
  )
  assert_false(
    broad_serving_pool_cancel_progress_valid(
      broad_serving_wb_cancel_boundary_sample(1UL, generated_tokens=3UL),
    ),
  )
}
