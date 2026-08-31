///|
const WORKER_GRAPH_TELEMETRY_BYTES : Int = 80

///|
const WORKER_GRAPH_TELEMETRY_MAGIC : UInt = 0x4758464c

///|
const WORKER_GRAPH_TELEMETRY_VERSION : UInt = 1

///|
const WORKER_GRAPH_TELEMETRY_KIND : UInt = 5

///|
const WORKER_GRAPH_TELEMETRY_CHECKSUM_OFFSET : Int = 72

///|
#valtype
pub(all) enum WorkerGraphExecutionPath {
  WorkerGraphCaptured
  WorkerGraphOrderedEager
} derive(Debug, Eq)

///|
/// Validated scalar graph shape carried across the private worker channel.
/// This is reporting data only and contains no launch or admission authority.
#valtype
pub struct WorkerGraphShapeClass {
  priv phase : Int
  priv batch_rows : Int
  priv query_tokens_per_row : Int
  priv context_tokens : Int
  priv token_rows : Int64
}

///|
/// Fixed-wire worker telemetry for the exact completed plan sequence.
#valtype
pub struct WorkerGraphTelemetry {
  priv sequence : UInt64
  priv path : Int
  priv shape_code : UInt64
  priv graph_hits : UInt64
  priv graph_misses : UInt64
}

///|
const WORKER_GRAPH_SHAPE_FIELD_MAX : Int = 0x1fffff

///|
fn worker_graph_shape_code(shape : WorkerGraphShapeClass) -> UInt64 {
  let phase = Int64::from_int(shape.phase).reinterpret_as_uint64()
  let batch = Int64::from_int(shape.batch_rows).reinterpret_as_uint64()
  let query = Int64::from_int(shape.query_tokens_per_row).reinterpret_as_uint64()
  let context = Int64::from_int(shape.context_tokens).reinterpret_as_uint64()
  phase | (batch << 1) | (query << 22) | (context << 43)
}

///|
fn worker_graph_shape_from_code(
  code : UInt64,
) -> WorkerGraphShapeClass raise WorkerWireError {
  if code == 0UL {
    raise InvalidFrame(Counts)
  }
  WorkerGraphShapeClass::new(
    phase=(code & 1UL).to_int(),
    batch_rows=((code >> 1) & 0x1fffffUL).to_int(),
    query_tokens_per_row=((code >> 22) & 0x1fffffUL).to_int(),
    context_tokens=((code >> 43) & 0x1fffffUL).to_int(),
    token_rows=Int64::from_int(((code >> 1) & 0x1fffffUL).to_int()) *
      Int64::from_int(((code >> 22) & 0x1fffffUL).to_int()),
  )
}

///|
pub fn WorkerGraphShapeClass::new(
  phase~ : Int,
  batch_rows~ : Int,
  query_tokens_per_row~ : Int,
  context_tokens~ : Int,
  token_rows~ : Int64,
) -> WorkerGraphShapeClass raise WorkerWireError {
  if (phase != 0 && phase != 1) ||
    batch_rows <= 0 ||
    query_tokens_per_row <= 0 ||
    context_tokens <= 0 ||
    batch_rows > WORKER_GRAPH_SHAPE_FIELD_MAX ||
    query_tokens_per_row > WORKER_GRAPH_SHAPE_FIELD_MAX ||
    context_tokens > WORKER_GRAPH_SHAPE_FIELD_MAX ||
    token_rows <= 0L ||
    Int64::from_int(batch_rows) * Int64::from_int(query_tokens_per_row) !=
    token_rows {
    raise InvalidFrame(Counts)
  }
  { phase, batch_rows, query_tokens_per_row, context_tokens, token_rows }
}

///|
fn worker_graph_telemetry(
  sequence~ : UInt64,
  path~ : WorkerGraphExecutionPath,
  shape_code~ : UInt64,
  graph_hits~ : UInt64,
  graph_misses~ : UInt64,
) -> WorkerGraphTelemetry raise WorkerWireError {
  if sequence == 0UL || (graph_hits == 0UL && graph_misses == 0UL) {
    raise InvalidFrame(Counts)
  }
  {
    sequence,
    path: match path {
      WorkerGraphCaptured => 1
      WorkerGraphOrderedEager => 2
    },
    shape_code,
    graph_hits,
    graph_misses,
  }
}

///|
pub fn WorkerGraphTelemetry::with_shape(
  sequence~ : UInt64,
  path~ : WorkerGraphExecutionPath,
  shape~ : WorkerGraphShapeClass,
  graph_hits~ : UInt64,
  graph_misses~ : UInt64,
) -> WorkerGraphTelemetry raise WorkerWireError {
  worker_graph_telemetry(
    sequence~,
    path~,
    shape_code=worker_graph_shape_code(shape),
    graph_hits~,
    graph_misses~,
  )
}

///|
pub fn WorkerGraphTelemetry::without_shape(
  sequence~ : UInt64,
  path~ : WorkerGraphExecutionPath,
  graph_hits~ : UInt64,
  graph_misses~ : UInt64,
) -> WorkerGraphTelemetry raise WorkerWireError {
  worker_graph_telemetry(
    sequence~,
    path~,
    shape_code=0UL,
    graph_hits~,
    graph_misses~,
  )
}

///|
pub fn worker_graph_telemetry_frame_capacity() -> Int {
  WORKER_GRAPH_TELEMETRY_BYTES
}

///|
/// Keeps graph-path diagnostics off the steady token-step transport. The
/// first four steps establish warmup behavior; later cumulative samples retain
/// visibility without adding a second IPC frame to every completion.
pub fn worker_graph_telemetry_sample_required(sequence : UInt64) -> Bool {
  sequence > 0UL && (sequence <= 4UL || sequence % 64UL == 0UL)
}

///|
pub fn WorkerGraphTelemetry::sequence_value(
  self : WorkerGraphTelemetry,
) -> UInt64 {
  self.sequence
}

///|
pub fn WorkerGraphTelemetry::selected_path(
  self : WorkerGraphTelemetry,
) -> WorkerGraphExecutionPath {
  if self.path == 1 {
    WorkerGraphCaptured
  } else {
    WorkerGraphOrderedEager
  }
}

///|
pub fn WorkerGraphTelemetry::has_selected_shape_class(
  self : WorkerGraphTelemetry,
) -> Bool {
  self.shape_code != 0UL
}

///|
pub fn WorkerGraphTelemetry::selected_shape_class(
  self : WorkerGraphTelemetry,
) -> WorkerGraphShapeClass raise WorkerWireError {
  worker_graph_shape_from_code(self.shape_code)
}

///|
pub fn WorkerGraphTelemetry::graph_hits(self : WorkerGraphTelemetry) -> UInt64 {
  self.graph_hits
}

///|
pub fn WorkerGraphTelemetry::graph_misses(
  self : WorkerGraphTelemetry,
) -> UInt64 {
  self.graph_misses
}

///|
pub fn WorkerGraphShapeClass::phase(self : WorkerGraphShapeClass) -> Int {
  self.phase
}

///|
pub fn WorkerGraphShapeClass::batch_rows(self : WorkerGraphShapeClass) -> Int {
  self.batch_rows
}

///|
pub fn WorkerGraphShapeClass::query_tokens_per_row(
  self : WorkerGraphShapeClass,
) -> Int {
  self.query_tokens_per_row
}

///|
pub fn WorkerGraphShapeClass::context_tokens(
  self : WorkerGraphShapeClass,
) -> Int {
  self.context_tokens
}

///|
pub fn WorkerGraphShapeClass::token_rows(self : WorkerGraphShapeClass) -> Int64 {
  self.token_rows
}

///|
fn worker_graph_path_to_wire(path : WorkerGraphExecutionPath) -> UInt {
  match path {
    WorkerGraphCaptured => 1
    WorkerGraphOrderedEager => 2
  }
}

///|
fn worker_graph_path_from_wire(
  value : UInt,
) -> WorkerGraphExecutionPath raise WorkerWireError {
  match value {
    1 => WorkerGraphCaptured
    2 => WorkerGraphOrderedEager
    _ => raise InvalidFrame(Kind)
  }
}

///|
fn worker_graph_telemetry_checksum(bytes : FixedArray[Byte]) -> UInt {
  let mut hash : UInt = 2166136261
  for index in 0..<WORKER_GRAPH_TELEMETRY_BYTES {
    if index < WORKER_GRAPH_TELEMETRY_CHECKSUM_OFFSET ||
      index >= WORKER_GRAPH_TELEMETRY_CHECKSUM_OFFSET + 4 {
      hash = (hash ^ bytes[index].to_uint()) * 16777619
    }
  }
  hash
}

///|
/// Encodes into caller-owned fixed storage. No completion/plan wire version is
/// changed; this is a separately versioned sidecar frame.
pub fn encode_worker_graph_telemetry(
  telemetry : WorkerGraphTelemetry,
  output : FixedArray[Byte],
) -> Int raise WorkerWireError {
  if output.length() < WORKER_GRAPH_TELEMETRY_BYTES {
    raise DestinationTooSmall
  }
  write_u32(output, 0, WORKER_GRAPH_TELEMETRY_MAGIC)
  write_u32(output, 4, WORKER_GRAPH_TELEMETRY_VERSION)
  write_u32(output, 8, WORKER_GRAPH_TELEMETRY_KIND)
  write_i32(output, 12, WORKER_GRAPH_TELEMETRY_BYTES)
  write_u64(output, 16, telemetry.sequence)
  write_u32(output, 24, worker_graph_path_to_wire(telemetry.selected_path()))
  if telemetry.shape_code == 0UL {
    write_u32(output, 28, 0)
    write_u32(output, 32, 0)
    write_i32(output, 36, 0)
    write_i32(output, 40, 0)
    write_i32(output, 44, 0)
    write_u64(output, 48, 0UL)
  } else {
    let shape = worker_graph_shape_from_code(telemetry.shape_code)
    write_u32(output, 28, 1)
    write_u32(output, 32, shape.phase().reinterpret_as_uint())
    write_i32(output, 36, shape.batch_rows())
    write_i32(output, 40, shape.query_tokens_per_row())
    write_i32(output, 44, shape.context_tokens())
    write_u64(output, 48, shape.token_rows().reinterpret_as_uint64())
  }
  write_u64(output, 56, telemetry.graph_hits)
  write_u64(output, 64, telemetry.graph_misses)
  write_u32(output, WORKER_GRAPH_TELEMETRY_CHECKSUM_OFFSET, 0)
  write_u32(output, 76, 0)
  write_u32(
    output,
    WORKER_GRAPH_TELEMETRY_CHECKSUM_OFFSET,
    worker_graph_telemetry_checksum(output),
  )
  WORKER_GRAPH_TELEMETRY_BYTES
}

///|
/// Validates an untrusted fixed sidecar before publishing scalar telemetry.
pub fn decode_worker_graph_telemetry(
  source : FixedArray[Byte],
  length : Int,
) -> WorkerGraphTelemetry raise WorkerWireError {
  if length != WORKER_GRAPH_TELEMETRY_BYTES || length > source.length() {
    raise InvalidFrame(Length)
  }
  if read_u32(source, 0) != WORKER_GRAPH_TELEMETRY_MAGIC {
    raise InvalidFrame(Magic)
  }
  if read_u32(source, 4) != WORKER_GRAPH_TELEMETRY_VERSION {
    raise InvalidFrame(Version)
  }
  if read_u32(source, 8) != WORKER_GRAPH_TELEMETRY_KIND {
    raise InvalidFrame(Kind)
  }
  if read_i32(source, 12) != WORKER_GRAPH_TELEMETRY_BYTES {
    raise InvalidFrame(Length)
  }
  if read_u32(source, 76) != 0 ||
    read_u32(source, WORKER_GRAPH_TELEMETRY_CHECKSUM_OFFSET) !=
    worker_graph_telemetry_checksum(source) {
    raise InvalidFrame(Checksum)
  }
  let sequence = read_u64(source, 16)
  let path = worker_graph_path_from_wire(read_u32(source, 24))
  let hits = read_u64(source, 56)
  let misses = read_u64(source, 64)
  match read_u32(source, 28) {
    0 => {
      if read_u32(source, 32) != 0 ||
        read_i32(source, 36) != 0 ||
        read_i32(source, 40) != 0 ||
        read_i32(source, 44) != 0 ||
        read_u64(source, 48) != 0UL {
        raise InvalidFrame(Reserved)
      }
      WorkerGraphTelemetry::without_shape(
        sequence~,
        path~,
        graph_hits=hits,
        graph_misses=misses,
      )
    }
    1 => {
      let shape = WorkerGraphShapeClass::new(
        phase=read_u32(source, 32).reinterpret_as_int(),
        batch_rows=read_i32(source, 36),
        query_tokens_per_row=read_i32(source, 40),
        context_tokens=read_i32(source, 44),
        token_rows=read_u64(source, 48).reinterpret_as_int64(),
      )
      WorkerGraphTelemetry::with_shape(
        sequence~,
        path~,
        shape~,
        graph_hits=hits,
        graph_misses=misses,
      )
    }
    _ => raise InvalidFrame(Reserved)
  }
}
