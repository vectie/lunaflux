///|
/// Offline measurements bind both the untuned and selected generated module.
/// This prevents observations from surviving a lowering change accidentally.
pub struct ProjectionFoldRecord {
  priv baseline : String
  priv artifact : String
  priv nanoseconds : Int64
  priv choices : ReadOnlyArray[@compiler.ProjectionFoldChoice]
}

///|
pub fn ProjectionFoldRecord::baseline_sha256(
  self : ProjectionFoldRecord,
) -> String {
  self.baseline
}

///|
pub fn ProjectionFoldRecord::artifact_sha256(
  self : ProjectionFoldRecord,
) -> String {
  self.artifact
}

///|
pub fn ProjectionFoldRecord::median_latency_ns(
  self : ProjectionFoldRecord,
) -> Int64 {
  self.nanoseconds
}

///|
pub fn ProjectionFoldRecord::fold_choices(
  self : ProjectionFoldRecord,
) -> ReadOnlyArray[@compiler.ProjectionFoldChoice] {
  self.choices
}

///|
fn fold_digest(value : StringView) -> Bool {
  value.length() == 64 && value.iter().all(c => c is ('0'..='9' | 'a'..='f'))
}

///|
/// Parse a bounded device/toolchain-scoped offline measurement table. `-`
/// denotes the baseline; other choices are role:stages:window:lookahead,
/// separated by commas. Roles are matrix, sibling, intermediate.
pub fn parse_projection_fold_records(
  bytes : Bytes,
  scope~ : String,
) -> ReadOnlyArray[ProjectionFoldRecord] raise ProjectionTuningError {
  guard bytes.length() <= 1048576 else { raise InvalidSnapshot }
  let text = @utf8.decode(bytes) catch { _ => raise InvalidSnapshot }
  let lines = text.trim_end().split("\n").to_array()
  guard lines is ["luna-projection-fold-v1", actual_scope, .. entries] &&
    actual_scope == "scope\t" + scope else {
    raise ScopeMismatch
  }
  guard entries.length() > 0 && entries.length() <= 256 else {
    raise InvalidSnapshot
  }
  let records : Array[ProjectionFoldRecord] = []
  for line in entries {
    guard line.split("\t").to_array()
      is ["record", baseline, artifact, ns, samples, encoded] &&
      fold_digest(baseline) &&
      fold_digest(artifact) else {
      raise InvalidSnapshot
    }
    let nanoseconds = decimal(ns)
    guard nanoseconds > 0L && integer(samples) >= 3 else {
      raise InvalidSnapshot
    }
    let choices : Array[@compiler.ProjectionFoldChoice] = []
    let roles : Array[@compiler.ProjectionFoldRole] = []
    if encoded != "-" {
      for item in encoded.split(",") {
        guard item.split(":").to_array() is [role, stages, window, fragments] else {
          raise InvalidSnapshot
        }
        let role : @compiler.ProjectionFoldRole = match role {
          "matrix" => MatrixFold
          "sibling" => SiblingFold
          "intermediate" => IntermediateFold
          _ => raise InvalidSnapshot
        }
        guard !roles.contains(role) else { raise InvalidSnapshot }
        roles.push(role)
        choices.push(
          @compiler.ProjectionFoldChoice::new(
            role,
            stages=integer(stages),
            transfer_tiles=integer(window),
            fragment_stages=integer(fragments),
          ) catch {
            _ => raise InvalidSnapshot
          },
        )
      }
    } else if baseline != artifact {
      raise InvalidSnapshot
    }
    for prior in records {
      guard prior.baseline[:] != baseline || prior.artifact[:] != artifact else {
        raise InvalidSnapshot
      }
    }
    records.push({
      baseline: baseline.to_owned(),
      artifact: artifact.to_owned(),
      nanoseconds,
      choices: ReadOnlyArray::from_array(choices),
    })
  }
  // Relative improvements require a matched baseline measurement.
  for record in records {
    guard records.any(r => r.baseline == record.baseline && r.choices.is_empty()) else {
      raise InvalidSnapshot
    }
  }
  ReadOnlyArray::from_array(records)
}
