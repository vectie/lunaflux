///|
/// A model execution boundary, independent of device lowering. A pipeline's
/// first stage owns embedding, its last stage owns output projection; a local
/// complete model owns both. Interior stages own neither.
pub(all) enum TextBoundary {
  Ingress
  Egress
  Complete
  Interior
} derive(Debug, Eq)

///|
pub extend TextBoundary with @debug.Debug::{to_repr}

///|
pub extend TextBoundary with Eq::{equal, not_equal}

///|
pub fn TextBoundary::has_ingress(self : Self) -> Bool {
  self is Ingress || self is Complete
}

///|
pub fn TextBoundary::has_egress(self : Self) -> Bool {
  self is Egress || self is Complete
}

///|
pub fn TextIoPrecision::boundary_workspace_bytes(
  self : Self,
  boundary : TextBoundary,
) -> Int64 {
  if boundary.has_egress() {
    self.workspace_bytes()
  } else {
    0L
  }
}
