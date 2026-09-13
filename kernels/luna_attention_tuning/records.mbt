///|
pub(all) suberror AttentionTuningError {
  InvalidTable
} derive(Debug, Eq)

///|
fn positive(text : StringView) -> Int64 raise AttentionTuningError {
  if text.is_empty() || !text.iter().all(c => c is ('0'..='9')) {
    raise InvalidTable
  }
  let value = @string.parse_int64(text) catch { _ => raise InvalidTable }
  if value <= 0L {
    raise InvalidTable
  }
  value
}

///|
/// Decode offline observations for one exact compiled frontier and explicit
/// benchmark workload vector. No observation is projected onto another source
/// frontier. The caller authenticates the file and supplies its device scope.
pub fn parse_attention_tuning_records(
  bytes : Bytes,
  scope~ : String,
  backend~ : @strategy.AttentionBackendId,
  target~ : @strategy.AttentionTargetId,
  problem~ : @strategy.AttentionCompileProblem,
  candidates : ArrayView[@strategy.AttentionTileCandidate],
) -> ReadOnlyArray[@strategy.AttentionAutotuneRecord] raise AttentionTuningError {
  if bytes.length() > 1048576 {
    raise InvalidTable
  }
  let text = @utf8.decode(bytes) catch { _ => raise InvalidTable }
  guard text.split("\n").to_array()
    is ["luna-attention-tuning-v1", actual_scope, .. lines, ""] &&
    actual_scope == "scope\t" + scope else {
    raise InvalidTable
  }
  let records : Array[@strategy.AttentionAutotuneRecord] = []
  let mut workloads = 0
  for line in lines {
    match line.split("\t").to_array() {
      ["workload", queries, rows, history] => {
        if !records.is_empty() {
          raise InvalidTable
        }
        let q = positive(queries)
        let r = positive(rows)
        let h = if history == "0" { 0L } else { positive(history) }
        if q < r || q > 1048576L || h > 1048576L {
          raise InvalidTable
        }
        workloads += 1
      }
      ["record", id, latency, samples] => {
        if workloads == 0 || records.length() >= 4096 {
          raise InvalidTable
        }
        let id = positive(id)
        let samples = positive(samples)
        if samples > 2147483647L {
          raise InvalidTable
        }
        guard candidates.filter(c => c.stable_id().to_int64() == id)
          is [candidate] else {
          raise InvalidTable
        }
        if records.any(r => r.key().candidate() == candidate) {
          raise InvalidTable
        }
        records.push(
          @strategy.AttentionAutotuneRecord::new(
            key=@strategy.AttentionAutotuneKey::new(
              backend~,
              target~,
              problem~,
              candidate~,
            ),
            median_latency_ns=positive(latency),
            sample_count=samples.to_int(),
          ) catch {
            _ => raise InvalidTable
          },
        )
      }
      _ => raise InvalidTable
    }
  }
  if records.is_empty() {
    raise InvalidTable
  }
  ReadOnlyArray::from_array(records)
}
