///|
fn LunaRequestPreparationLane::require_request(
  self : LunaRequestPreparationLane,
) -> @inference.GenerateRequest {
  match self.request {
    Some(value) => value
    None => abort("Luna preparation request invariant")
  }
}

///|
fn LunaRequestPreparationLane::require_deadline(
  self : LunaRequestPreparationLane,
) -> @inference.AdmissionDeadline {
  match self.deadline {
    Some(value) => value
    None => abort("Luna preparation deadline invariant")
  }
}

///|
fn LunaRequestPreparationLane::require_effective(
  self : LunaRequestPreparationLane,
) -> @inference.EffectiveLimits {
  match self.effective {
    Some(value) => value
    None => abort("Luna preparation effective-limit invariant")
  }
}
