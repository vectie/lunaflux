///|
/// Offline complete-chain latency for one representative execution shape.
/// Candidate IDs identify backend-owned immutable launch lists, not devices.
pub struct ExecutionGraphRouteMeasurement {
  priv shape : ExecutionGraphShape
  priv candidate : Int
  priv median_ns : Int64
  priv samples : Int
}

///|
pub fn ExecutionGraphRouteMeasurement::new(
  shape : ExecutionGraphShape,
  candidate~ : Int,
  median_ns~ : Int64,
  samples~ : Int,
) -> ExecutionGraphRouteMeasurement raise ExecutionGraphBucketError {
  if candidate < 0 || median_ns <= 0L || samples < 3 {
    raise InvalidExecutionGraphBucket(Shape)
  }
  { shape, candidate, median_ns, samples, }
}

///|
/// Immutable startup result. Missing observations retain the caller's normal
/// route, rather than generalizing a winner from another context or phase.
pub struct ExecutionGraphMeasuredRoutes {
  priv buckets : ExecutionGraphBucketTable
  priv candidates : ReadOnlyArray[Int]
}

///|
/// Every measured bucket must include its baseline. An observation represents
/// the same workload vector for every candidate in that bucket; the importer
/// is responsible for checking device/source scope and vector identity.
pub fn ExecutionGraphMeasuredRoutes::new(
  buckets : ExecutionGraphBucketTable,
  baseline~ : Int,
  observations : ArrayView[ExecutionGraphRouteMeasurement],
  minimum_improvement_basis_points? : Int = 100,
) -> ExecutionGraphMeasuredRoutes raise ExecutionGraphBucketError {
  if baseline < 0 ||
    observations.length() > 4096 ||
    minimum_improvement_basis_points < 0 ||
    minimum_improvement_basis_points > 10000 {
    raise InvalidExecutionGraphBucket(Limit)
  }
  let slots = Array::makei(observations.length(), i => {
    buckets.select(observations[i].shape).slot()
  })
  let best = Array::make(buckets.bucket_count(), -1)
  let baseline_times = FixedArray::make(buckets.bucket_count(), 0L)
  let baseline_samples = FixedArray::make(buckets.bucket_count(), 0)
  let baseline_shapes : FixedArray[ExecutionGraphShape?] = FixedArray::make(
    buckets.bucket_count(),
    None,
  )
  let best_times = FixedArray::make(buckets.bucket_count(), 0L)
  for index, observation in observations {
    for prior in 0..<index {
      if slots[prior] == slots[index] &&
        observations[prior].candidate == observation.candidate {
        // Multiple representatives require aggregation by the importer; never
        // silently count one candidate's extra measurements as extra weight.
        raise InvalidExecutionGraphBucket(Shape)
      }
    }
    if observation.candidate == baseline {
      baseline_times[slots[index]] = observation.median_ns
      baseline_samples[slots[index]] = observation.samples
      baseline_shapes[slots[index]] = Some(observation.shape)
      best_times[slots[index]] = observation.median_ns
      best[slots[index]] = baseline
    }
  }
  for index, observation in observations {
    let slot = slots[index]
    let base = baseline_times[slot]
    if base == 0L || baseline_samples[slot] != observation.samples {
      raise InvalidExecutionGraphBucket(Shape)
    }
    guard baseline_shapes[slot] is Some(shape) &&
      shape.phase == observation.shape.phase &&
      shape.batch_rows == observation.shape.batch_rows &&
      shape.query_tokens == observation.shape.query_tokens &&
      shape.context_tokens == observation.shape.context_tokens else {
      raise InvalidExecutionGraphBucket(Shape)
    }
    let margin = minimum_improvement_basis_points.to_int64()
    // ceil(base*margin/10000), without an overflowing product.
    let required = base / 10000L * margin +
      (base % 10000L * margin + 9999L) / 10000L
    if observation.candidate != baseline &&
      observation.median_ns <= base - required &&
      (
        observation.median_ns < best_times[slot] ||
        (
          observation.median_ns == best_times[slot] &&
          best[slot] != baseline &&
          observation.candidate < best[slot]
        )
      ) {
      best[slot] = observation.candidate
      best_times[slot] = observation.median_ns
    }
  }
  { buckets, candidates: ReadOnlyArray::from_array(best), }
}

///|
/// Bounded arithmetic and one immutable lookup; no scanning or allocation.
pub fn ExecutionGraphMeasuredRoutes::select(
  self : ExecutionGraphMeasuredRoutes,
  shape : ExecutionGraphShape,
) -> Int? raise ExecutionGraphBucketError {
  let candidate = self.candidates[self.buckets.select(shape).slot()]
  if candidate < 0 {
    None
  } else {
    Some(candidate)
  }
}
