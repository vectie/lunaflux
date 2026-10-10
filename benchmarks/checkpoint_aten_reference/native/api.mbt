///|
pub suberror ReferenceError {
  Native(String)
  ClosedOwner
}

///|
pub struct Session {
  raw : RawSession
  mut closed : Bool
}

///|
pub struct Tensor {
  raw : RawTensor
  session : Session
  mut closed : Bool
}

///|
pub(all) enum Op {
  RmsNorm
  Linear
  Cosine
  Sine
  Rotary
  Attention
  SiluMultiply
  Add
  ConcatRows
  Constant
}

///|
pub fn Session::new(
  library : String,
  device~ : Int,
  budget~ : Int64,
) -> Self raise ReferenceError {
  let raw = session_open(@utf8.encode(library), device, budget)
  if !session_valid(raw) {
    let message = @utf8.decode_lossy(session_error(raw))
    session_close(raw)
    raise Native(message)
  }
  { raw, closed: false, }
}

///|
pub fn Session::close(self : Self) -> Unit {
  if !self.closed {
    self.closed = true
    session_close(self.raw)
  }
}

///|
pub fn Tensor::close(self : Self) -> Unit {
  if !self.closed {
    self.closed = true
    if !self.session.closed {
      tensor_close(self.session.raw, self.raw)
    }
  }
}

///|
fn Session::require_open(self : Self) -> Unit raise ReferenceError {
  guard !self.closed else { raise ClosedOwner }
}

///|
fn Session::owned(self : Self, raw : RawTensor) -> Tensor raise ReferenceError {
  guard tensor_valid(raw) else {
    raise Native(@utf8.decode_lossy(session_error(self.raw)))
  }
  { raw, session: self, closed: false, }
}

///|
fn Session::operand(
  self : Self,
  tensor : Tensor?,
) -> RawTensor raise ReferenceError {
  match tensor {
    None => tensor_null()
    Some(t) => {
      guard !t.closed &&
        !t.session.closed &&
        session_same(t.session.raw, self.raw) else {
        raise ClosedOwner
      }
      t.raw
    }
  }
}

///|
pub fn Session::bytes(
  self : Self,
  path : String,
  offset? : Int64 = 0L,
  length? : Int = -1,
) -> Bytes raise ReferenceError {
  self.require_open()
  let bytes = read_file(self.raw, @utf8.encode(path), offset, length)
  let error = session_error(self.raw)
  guard error.is_empty() else { raise Native(@utf8.decode_lossy(error)) }
  bytes
}

///|
pub fn Session::load(
  self : Self,
  path : String,
  offset : Int64,
  rows : Int,
  columns : Int,
) -> Tensor raise ReferenceError {
  self.require_open()
  self.owned(load_bf16(self.raw, @utf8.encode(path), offset, rows, columns))
}

///|
pub fn Session::run(
  self : Self,
  op : Op,
  x? : Tensor,
  y? : Tensor,
  z? : Tensor,
  a? : Int = 0,
  b? : Int = 0,
  c? : Int = 0,
  d? : Double = 0.0,
) -> Tensor raise ReferenceError {
  self.require_open()
  let code = match op {
    RmsNorm => 0
    Linear => 1
    Cosine => 2
    Sine => 3
    Rotary => 4
    Attention => 5
    SiluMultiply => 6
    Add => 7
    ConcatRows => 8
    Constant => 9
  }
  self.owned(
    apply(
      self.raw,
      code,
      self.operand(x),
      self.operand(y),
      self.operand(z),
      a,
      b,
      c,
      d,
    ),
  )
}

///|
pub fn Session::retained_bytes(self : Self) -> Int64 raise ReferenceError {
  self.require_open()
  live_bytes(self.raw)
}

///|
pub fn Tensor::write(self : Self, path : String) -> Unit raise ReferenceError {
  guard !self.closed && !self.session.closed else { raise ClosedOwner }
  guard write_bf16(self.session.raw, self.raw, @utf8.encode(path)) == 0 else {
    raise Native(@utf8.decode_lossy(session_error(self.session.raw)))
  }
}
