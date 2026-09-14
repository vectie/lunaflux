///|
/// Source/device-scoped offline observations retain the representative shape
/// for each candidate instead of collapsing every workload into one latency.
/// The returned immutable selector is built at startup, never in a token step.
pub fn parse_attention_bucket_routes(
  bytes : Bytes,
  scope~ : String,
  buckets : @graph.ExecutionGraphBucketTable,
  baseline~ : Int,
  admitted_candidates : ArrayView[Int],
) -> @graph.ExecutionGraphMeasuredRoutes raise AttentionTuningError {
  if bytes.length() > 1048576 ||
    scope.is_empty() ||
    admitted_candidates.is_empty() ||
    admitted_candidates.length() > 256 ||
    !admitted_candidates.contains(baseline) {
    raise InvalidTable
  }
  for index, candidate in admitted_candidates {
    if candidate <= 0 || admitted_candidates[:index].contains(candidate) {
      raise InvalidTable
    }
  }
  let text = @utf8.decode(bytes) catch { _ => raise InvalidTable }
  guard text.split("\n").to_array()
    is ["luna-attention-buckets-v1", actual, .. lines, ""] &&
    actual == "scope\t" + scope else {
    raise InvalidTable
  }
  let observations : Array[@graph.ExecutionGraphRouteMeasurement] = []
  for line in lines {
    guard line.split("\t").to_array()
      is ["record", phase, rows, queries, context, candidate, latency, samples] else {
      raise InvalidTable
    }
    if observations.length() >= 4096 {
      raise InvalidTable
    }
    let phase : @graph.ExecutionGraphPhase = match phase {
      "prefill" => PrefillGraph
      "decode" => DecodeGraph
      "mixed" => MixedGraph
      _ => raise InvalidTable
    }
    let rows = positive(rows)
    let queries = positive(queries)
    let context = positive(context)
    let candidate = positive(candidate)
    let samples = positive(samples)
    if rows > queries ||
      queries > 1048576L ||
      context > 1048576L ||
      candidate > 2147483647L ||
      samples > 2147483647L ||
      !admitted_candidates.contains(candidate.to_int()) ||
      (phase == DecodeGraph && rows != queries) ||
      (phase == MixedGraph && rows < 2L) {
      raise InvalidTable
    }
    let shape = @graph.ExecutionGraphShape::new(
      phase~,
      batch_rows=rows.to_int(),
      query_tokens=queries.to_int(),
      context_tokens=context.to_int(),
    ) catch {
      _ => raise InvalidTable
    }
    observations.push(
      @graph.ExecutionGraphRouteMeasurement::new(
        shape,
        candidate=candidate.to_int(),
        median_ns=positive(latency),
        samples=samples.to_int(),
      ) catch {
        _ => raise InvalidTable
      },
    )
  }
  if observations.is_empty() {
    raise InvalidTable
  }
  @graph.ExecutionGraphMeasuredRoutes::new(buckets, baseline~, observations[:]) catch {
    _ => raise InvalidTable
  }
}
