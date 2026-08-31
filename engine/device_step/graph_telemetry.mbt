///|
/// Actual execution base selected once by the private ordered executor during
/// startup. A captured completion is a graph hit; an ordered-eager completion
/// is a graph miss.
pub(all) enum PagedGraphSelectedPath {
  PagedGraphCaptured
  PagedGraphOrderedEager
} derive(Debug, Eq)

///|
/// Scalar copy of the authenticated Luna manifest shape selected at startup.
/// The private fields prevent a caller from attaching an asserted shape to an
/// executor telemetry report.
#valtype
pub struct PagedGraphSelectedShapeClass {
  priv phase : @device_profile.ExecutionPhase
  priv batch_rows : Int
  priv query_tokens_per_row : Int
  priv context_tokens : Int
  priv token_rows : Int64
}

///|
/// Allocation-free scalar snapshot of one executor's runtime graph telemetry.
/// It carries neither mutable telemetry ownership nor device/scheduler
/// authority.
#valtype
pub struct PagedGraphRuntimeTelemetry {
  priv selected_path : PagedGraphSelectedPath
  priv selected_shape : PagedGraphSelectedShapeClass?
  priv graph_hits : UInt64
  priv graph_misses : UInt64
}

///|
priv struct PagedGraphTelemetryOwner {
  selected_path : PagedGraphSelectedPath
  selected_shape : PagedGraphSelectedShapeClass?
  mut graph_hits : UInt64
  mut graph_misses : UInt64
}

///|
fn selected_graph_shape_class(
  shape : @luna_capability_manifest.LunaGraphShapeClass?,
) -> PagedGraphSelectedShapeClass? {
  match shape {
    None => None
    Some(shape) =>
      Some({
        phase: shape.phase(),
        batch_rows: shape.batch_rows(),
        query_tokens_per_row: shape.query_tokens_per_row(),
        context_tokens: shape.context_tokens(),
        token_rows: shape.token_rows(),
      })
  }
}

///|
fn new_paged_graph_telemetry(
  mode : @device.OrderedKernelExecutionMode,
  shape : @luna_capability_manifest.LunaGraphShapeClass?,
) -> PagedGraphTelemetryOwner {
  {
    selected_path: match mode {
      OrderedKernelCaptured => PagedGraphCaptured
      OrderedKernelEager => PagedGraphOrderedEager
    },
    selected_shape: selected_graph_shape_class(shape),
    graph_hits: 0UL,
    graph_misses: 0UL,
  }
}

///|
fn PagedGraphTelemetryOwner::record_completed(
  self : PagedGraphTelemetryOwner,
  mode : @device.OrderedKernelExecutionMode,
) -> Unit {
  match mode {
    OrderedKernelCaptured =>
      if self.graph_hits != 0xffffffffffffffffUL {
        self.graph_hits += 1UL
      }
    OrderedKernelEager =>
      if self.graph_misses != 0xffffffffffffffffUL {
        self.graph_misses += 1UL
      }
  }
}

///|
fn PagedGraphTelemetryOwner::snapshot(
  self : PagedGraphTelemetryOwner,
) -> PagedGraphRuntimeTelemetry {
  {
    selected_path: self.selected_path,
    selected_shape: self.selected_shape,
    graph_hits: self.graph_hits,
    graph_misses: self.graph_misses,
  }
}

///|
pub fn PagedGraphRuntimeTelemetry::selected_path(
  self : PagedGraphRuntimeTelemetry,
) -> PagedGraphSelectedPath {
  self.selected_path
}

///|
pub fn PagedGraphRuntimeTelemetry::selected_shape_class(
  self : PagedGraphRuntimeTelemetry,
) -> PagedGraphSelectedShapeClass? {
  self.selected_shape
}

///|
pub fn PagedGraphRuntimeTelemetry::graph_hits(
  self : PagedGraphRuntimeTelemetry,
) -> UInt64 {
  self.graph_hits
}

///|
pub fn PagedGraphRuntimeTelemetry::graph_misses(
  self : PagedGraphRuntimeTelemetry,
) -> UInt64 {
  self.graph_misses
}

///|
pub fn PagedGraphSelectedShapeClass::phase(
  self : PagedGraphSelectedShapeClass,
) -> @device_profile.ExecutionPhase {
  self.phase
}

///|
pub fn PagedGraphSelectedShapeClass::batch_rows(
  self : PagedGraphSelectedShapeClass,
) -> Int {
  self.batch_rows
}

///|
pub fn PagedGraphSelectedShapeClass::query_tokens_per_row(
  self : PagedGraphSelectedShapeClass,
) -> Int {
  self.query_tokens_per_row
}

///|
pub fn PagedGraphSelectedShapeClass::context_tokens(
  self : PagedGraphSelectedShapeClass,
) -> Int {
  self.context_tokens
}

///|
pub fn PagedGraphSelectedShapeClass::token_rows(
  self : PagedGraphSelectedShapeClass,
) -> Int64 {
  self.token_rows
}

///|
/// Returns the current scalar counters and the immutable startup selection.
/// Reporting remains available after poison/close for terminal diagnostics.
pub fn PagedGraphExecutor::graph_runtime_telemetry(
  self : PagedGraphExecutor,
) -> PagedGraphRuntimeTelemetry {
  self.graph_telemetry.snapshot()
}
