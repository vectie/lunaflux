///|
fn eligible(
  candidate : Candidate,
  budget : ResourceBudget,
  registers : Int?,
  residency : Int?,
) -> Bool {
  candidate.threads <= budget.threads &&
  candidate.shared <= budget.shared &&
  (match (budget.registers, registers) {
    (None, _) => true
    (Some(limit), Some(value)) => value <= limit
    _ => false
  }) &&
  (match (budget.residency, residency) {
    (None, _) => true
    (Some(limit), Some(value)) => value >= limit
    _ => false
  })
}

///|
/// Pure, bounded offline selection. Measurements must match device, toolchain,
/// shape, program and generated artifact. Ambiguous duplicate identities are
/// rejected. Equal timings use the policy ID, never input iteration order.
/// No observations means explicitly unmeasured preference, not an optimality
/// claim. Backend candidate enumeration determines supported instruction paths.
pub fn select(
  scope : Scope,
  budget : ResourceBudget,
  candidates : ArrayView[Candidate],
  observations : ArrayView[Observation],
  preferred_id~ : String,
) -> Selection? {
  guard !candidates.is_empty() &&
    candidates.length() <= 256 &&
    observations.length() <= 4096 else {
    return None
  }
  for i in 0..<candidates.length() {
    for j in 0..<i {
      if candidates[i].policy.id == candidates[j].policy.id {
        return None
      }
    }
  }
  for i in 0..<observations.length() {
    let record = observations[i]
    if record.scope != scope {
      continue
    }
    for j in 0..<i {
      let prior = observations[j]
      if prior.scope == scope &&
        prior.policy_id == record.policy_id &&
        prior.artifact == record.artifact {
        return None
      }
    }
  }
  let mut measured : Selection? = None
  let mut fallback : Candidate? = None
  let mut preferred : Candidate? = None
  for candidate in candidates {
    let observation = observations
      .iter()
      .find_first(r => {
        r.scope == scope &&
        r.policy_id == candidate.policy.id &&
        r.artifact == candidate.artifact
      })
    if observation is Some(record) {
      // A resource-ineligible measured candidate is not reconsidered using an
      // optimistic unmeasured compilation bound.
      if !eligible(
          candidate,
          budget,
          Some(record.registers),
          Some(record.residency),
        ) {
        continue
      }
      let better = match measured {
        None => true
        Some({ origin: Measured(latency_ns=ns, ..), candidate: old, }) =>
          record.latency < ns ||
          (record.latency == ns && candidate.policy.id < old.policy.id)
        _ => false
      }
      if better {
        measured = Some({
          candidate,
          origin: Measured(latency_ns=record.latency, samples=record.samples),
        })
      }
    } else if eligible(
        candidate,
        budget,
        candidate.registers,
        candidate.residency,
      ) {
      if candidate.policy.id == preferred_id {
        preferred = Some(candidate)
      }
      if fallback is None ||
        (fallback is Some(old) && candidate.policy.id < old.policy.id) {
        fallback = Some(candidate)
      }
    }
  }
  match (measured, preferred, fallback) {
    (Some(selection), _, _) => Some(selection)
    (_, Some(candidate), _) => Some({ candidate, origin: UnmeasuredPreferred, })
    (_, _, Some(candidate)) => Some({ candidate, origin: UnmeasuredEligible, })
    _ => None
  }
}
