///|
fn local_benchmark_request_contract(
  owner : RuntimeInstanceOwner,
  release : RuntimeReleaseAdmission,
) -> RuntimeSpawnedBenchmarkContract raise RuntimeLocalBenchmarkError {
  if owner.service_protocol() != RuntimeNativeFramedV1 ||
    owner.preparations.length() != 1 ||
    owner.phase() != HealthyNotReady {
    raise local_benchmark_error(LocalBenchmarkContract)
  }
  let preparation = owner.preparations[0]
  let limits = preparation.inference_limits() catch {
    _ => raise local_benchmark_error(LocalBenchmarkContract)
  }
  let identity = preparation.model_identity() catch {
    _ => raise local_benchmark_error(LocalBenchmarkContract)
  }
  if identity.content().as_hex() != release.model_content_sha256() ||
    identity.plan().as_hex() != release.model_plan_sha256() ||
    limits.max_input_tokens() < 1 ||
    limits.max_new_tokens() < 2 ||
    limits.max_context_tokens() < 3 ||
    limits.max_token_id() < 0 ||
    limits.max_cache_scope_bytes() < 1 ||
    limits.max_deadline_millis() == 0UL {
    raise local_benchmark_error(LocalBenchmarkContract)
  }
  let timeout = if limits.max_deadline_millis() > 5000UL {
    5000UL
  } else {
    limits.max_deadline_millis()
  }
  let request = @inference.GenerateRequest::new(
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(1UL),
    model_identity=identity,
    input=@inference.Input::from_token_ids([0], limits),
    max_new_tokens=2,
    context_ceiling=3,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(0UL),
    stops=@inference.StopConditions::new([], [], limits),
    stream_preference=Streaming,
    deadline=@inference.DeadlineBudget::from_milliseconds(timeout, limits),
    cache=@inference.CachePolicy::new(
      @inference.CacheScope::from_ascii(b"b", limits),
      Disabled,
    ),
    trace=None,
    limits~,
  ) catch {
    _ => raise local_benchmark_error(LocalBenchmarkContract)
  }
  let framed = preparation.framed_limits() catch {
    _ => raise local_benchmark_error(LocalBenchmarkContract)
  }
  let encoded = @framed_wire.RequestFrameBuffer::new(framed).encode(request) catch {
    _ => raise local_benchmark_error(LocalBenchmarkContract)
  }
  let encoded_length = encoded.length() catch {
    _ => raise local_benchmark_error(LocalBenchmarkContract)
  }
  let frame = FixedArray::make(encoded_length, b'\x00')
  ignore(
    encoded.copy_to(frame, destination_offset=0) catch {
      _ => raise local_benchmark_error(LocalBenchmarkContract)
    },
  )
  let serving = RuntimeSpawnedServingValidationContract::new(
    frame,
    identity,
    0,
    0,
    1,
    3,
    3,
    limits,
    framed.max_frame_bytes(),
    1048576,
    timeout.to_int(),
  ) catch {
    _ => raise local_benchmark_error(LocalBenchmarkContract)
  }
  {
    serving,
    profile: Latency,
    ordinal: 1,
    order_position: 0,
    verify_expected_tokens: false,
  }
}

///|
/// Authenticates the same digest-suffixed deployment grammar as `run`, starts
/// the real native owner, measures one internally fixed request with the
/// monotonic collector, and returns evidence only after complete drain/close.
/// No caller timestamp, duration, comparison, or promotion input is accepted.
pub async fn benchmark_local_deployment(
  model_argument : StringView,
  limits : RuntimeInstanceFileLimits,
) -> RuntimeLocalBenchmarkEvidence {
  let owner = prepare(model_argument, limits) catch {
    _ => raise local_benchmark_error(LocalBenchmarkRelease)
  }
  // This receipt was derived during `prepare` from the exact admitted inputs
  // and executable consumed by `owner`; there is no second deployment parse.
  let release = owner.release_admission
  let contract = local_benchmark_request_contract(owner, release) catch {
    _ => {
      let cleaned = drain_spawned_serving_owner(owner, 1048576)
      if !cleaned {
        raise local_benchmark_error(LocalBenchmarkCleanup)
      }
      raise local_benchmark_error(LocalBenchmarkContract)
    }
  }
  let result = owner.benchmark_spawned_serving_request(contract) catch {
    _ => raise local_benchmark_error(LocalBenchmarkRuntime)
  }
  if owner.phase() != Closed ||
    owner.readiness() != RuntimeNotReady ||
    !owner.cleanup_complete() {
    raise local_benchmark_error(LocalBenchmarkCleanup)
  }
  local_benchmark_evidence(release, result)
}
