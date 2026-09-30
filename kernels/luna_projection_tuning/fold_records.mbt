///|
/// Offline measurements bind both the untuned and selected generated module.
/// This prevents observations from surviving a lowering change accidentally.
pub struct ProjectionFoldRecord {
  priv baseline : String
  priv artifact : String
  priv nanoseconds : Int64
  priv samples : Int
  priv choices : ReadOnlyArray[@compiler.ProjectionFoldChoice]
  priv sibling_ownership : @compiler.ProjectionSiblingOwnership
  priv sibling_transport : @compiler.ProjectionSiblingTransport
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
pub fn ProjectionFoldRecord::sample_count(self : ProjectionFoldRecord) -> Int {
  self.samples
}

///|
pub fn ProjectionFoldRecord::fold_choices(
  self : ProjectionFoldRecord,
) -> ReadOnlyArray[@compiler.ProjectionFoldChoice] {
  self.choices
}

///|
pub fn ProjectionFoldRecord::sibling_ownership(
  self : ProjectionFoldRecord,
) -> @compiler.ProjectionSiblingOwnership {
  self.sibling_ownership
}

///|
pub fn ProjectionFoldRecord::is_baseline(self : ProjectionFoldRecord) -> Bool {
  self.choices.is_empty() &&
  self.sibling_ownership == SplitSiblingPlanes &&
  self.sibling_transport == BaselineSiblingTransport
}

///|
pub fn ProjectionFoldRecord::sibling_transport(
  self : ProjectionFoldRecord,
) -> @compiler.ProjectionSiblingTransport {
  self.sibling_transport
}

///|
fn fold_digest(value : StringView) -> Bool {
  value.length() == 64 && value.iter().all(c => c is ('0'..='9' | 'a'..='f'))
}

///|
/// Parse a bounded device/toolchain-scoped offline measurement table. `-`
/// denotes the baseline; other choices are role:stages:window:lookahead,
/// separated by commas. V2 appends row_tiles:output_tiles:hoist, where zero
/// geometry retains the ordinary schedule. Roles are matrix, sibling, intermediate.
pub fn parse_projection_fold_records(
  bytes : Bytes,
  scope~ : String,
) -> ReadOnlyArray[ProjectionFoldRecord] raise ProjectionTuningError {
  guard bytes.length() <= 1048576 else { raise InvalidSnapshot }
  let text = @utf8.decode(bytes) catch { _ => raise InvalidSnapshot }
  let lines = text.trim_end().split("\n").to_array()
  guard lines is [version, actual_scope, .. entries] &&
    (
      version == "luna-projection-fold-v1" ||
      version == "luna-projection-fold-v2" ||
      version == "luna-projection-fold-v3"
    ) &&
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
    let sample_count = integer(samples)
    guard nanoseconds > 0L && sample_count >= 3 else { raise InvalidSnapshot }
    let choices : Array[@compiler.ProjectionFoldChoice] = []
    let roles : Array[(@compiler.ProjectionFoldRole, Int?)] = []
    let mut sibling_ownership : @compiler.ProjectionSiblingOwnership = SplitSiblingPlanes
    let mut has_ownership = false
    let mut sibling_transport : @compiler.ProjectionSiblingTransport = BaselineSiblingTransport
    let mut has_transport = false
    if encoded != "-" {
      for item in encoded.split(",") {
        if item.split(":").to_array() is ["transport", transport] {
          guard version != "luna-projection-fold-v1" && !has_transport else {
            raise InvalidSnapshot
          }
          sibling_transport = match transport {
            "segmented" => SegmentedSiblingTransport
            "independent" => IndependentSiblingTransport
            _ => raise InvalidSnapshot
          }
          has_transport = true
          continue
        }
        if item.split(":").to_array() is ["epilogue", ownership] {
          guard version != "luna-projection-fold-v1" && !has_ownership else {
            raise InvalidSnapshot
          }
          sibling_ownership = match ownership {
            "retained" => RetainSiblingProducer
            "coowned" => CoownedSiblingValues
            _ => raise InvalidSnapshot
          }
          has_ownership = true
          continue
        }
        let (
          role,
          stages,
          window,
          fragments,
          row_tiles,
          output_tiles,
          hoist,
          consumer_groups,
          maximum_token_rows,
        ) = match item.split(":").to_array() {
          [role, stages, window, fragments] if version ==
            "luna-projection-fold-v1" =>
            (role, stages, window, fragments, None, None, false, None, None)
          [role, stages, window, fragments, rows, columns, addresses, .. groups] if version ==
            "luna-projection-fold-v2" ||
            version == "luna-projection-fold-v3" => {
            let rows = integer(rows)
            let columns = integer(columns)
            guard addresses == "0" || addresses == "1" else {
              raise InvalidSnapshot
            }
            let (groups, bound) = match groups {
              [] => (None, None)
              [groups] => (Some(integer(groups)), None)
              [groups, bound] if version == "luna-projection-fold-v3" =>
                (Some(integer(groups)), Some(integer(bound)))
              _ => raise InvalidSnapshot
            }
            (
              role,
              stages,
              window,
              fragments,
              if rows == 0 {
                None
              } else {
                Some(rows)
              },
              if columns == 0 {
                None
              } else {
                Some(columns)
              },
              addresses == "1",
              groups,
              bound,
            )
          }
          _ => raise InvalidSnapshot
        }
        let role : @compiler.ProjectionFoldRole = match role {
          "matrix" => MatrixFold
          "sibling" => SiblingFold
          "intermediate" => IntermediateFold
          _ => raise InvalidSnapshot
        }
        let key = (role, maximum_token_rows)
        guard !roles.contains(key) else { raise InvalidSnapshot }
        roles.push(key)
        choices.push(
          @compiler.ProjectionFoldChoice::new(
            role,
            stages=integer(stages),
            transfer_tiles=integer(window),
            fragment_stages=integer(fragments),
            row_tiles?,
            output_tiles?,
            hoist_addresses=hoist,
            consumer_groups?,
            maximum_token_rows?,
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
      samples: sample_count,
      choices: ReadOnlyArray::from_array(choices),
      sibling_ownership,
      sibling_transport,
    })
  }
  // Relative improvements require a matched baseline measurement.
  for record in records {
    guard records.any(r => {
      r.baseline == record.baseline &&
      r.is_baseline() &&
      r.samples == record.samples
    }) else {
      raise InvalidSnapshot
    }
  }
  ReadOnlyArray::from_array(records)
}
