///|
pub fn BatchedDecoderRank::begin(
  self : Self,
  plan : @wire.ValidatedPlanFrame,
  writer : @wire.CompletionFrameWriter,
) -> Unit raise {
  guard self.state is Ready else { raise @frame.BatchFrameError::Busy }
  self.frame.stage(plan, writer)
  self.state = Failed
  for index in 0..<self.ports.length() {
    let port = self.ports[index]
    let bytes = self.frame.copy_input(port, self.payloads[index])
    self.live_bytes[index] = bytes
    if bytes == 0 {
      continue
    }
    let pool = self.pools[index]
    pool.write_host(
      self.payloads[index],
      host_offset=0L,
      source_offset=0,
      bytes~,
    )
    pool.enqueue(
      lane=0,
      host_offset=0L,
      device_offset=0L,
      bytes=bytes.to_int64(),
      to_device=true,
    )
    pool.record(0)
  }
  self.state = Uploading
}

///|
pub fn BatchedDecoderRank::inputs_ready(self : Self) -> Bool raise {
  if self.state is InputsReady {
    return true
  }
  guard self.state is Uploading else { raise @frame.BatchFrameError::Busy }
  self.state = Failed
  for index in 0..<self.ports.length() {
    if self.live_bytes[index] > 0 && !self.pools[index].poll(0) {
      self.state = Uploading
      return false
    }
  }
  for index in 0..<self.ports.length() {
    if self.live_bytes[index] > 0 {
      self.pools[index].reset(0)
    }
  }
  self.state = InputsReady
  true
}

///|
pub fn BatchedDecoderRank::submit(self : Self) -> Unit raise {
  guard self.state is InputsReady && self.effects is Some(effects) else {
    raise @frame.BatchFrameError::Busy
  }
  self.state = Failed
  (effects.submit)(self.frame.live_rows())
  self.state = Executing
}

///|
fn load_i32(bytes : FixedArray[Byte], row : Int) -> Int {
  for i in 0..<4; bits = 0U {
    continue bits | (bytes[row * 4 + i].to_uint() << (i * 8))
  } nobreak {
    bits.reinterpret_as_int()
  }
}

///|
fn BatchedDecoderRank::download_bytes(self : Self, index : Int) -> Int raise {
  if index == self.ports.length() {
    if self.outputs_token {
      self.frame.output_rows() * 4
    } else {
      0
    }
  } else {
    4
  }
}

///|
pub fn BatchedDecoderRank::progress(self : Self) -> Bool raise {
  guard self.effects is Some(effects) else {
    raise @frame.BatchFrameError::Busy
  }
  match self.state {
    Executing => {
      self.state = Failed
      if !(effects.poll)() {
        self.state = Executing
        return false
      }
      for index in self.ports.length()..<self.pools.length() {
        let bytes = self.download_bytes(index)
        if bytes == 0 {
          continue
        }
        let pool = self.pools[index]
        pool.enqueue(
          lane=0,
          host_offset=0L,
          device_offset=if index == self.ports.length() { 0L } else { 8L },
          bytes=bytes.to_int64(),
          to_device=false,
        )
        pool.record(0)
      }
      self.state = Downloading
      false
    }
    Downloading => {
      self.state = Failed
      for index in self.ports.length()..<self.pools.length() {
        if self.download_bytes(index) > 0 && !self.pools[index].poll(0) {
          self.state = Downloading
          return false
        }
      }
      self.failed = false
      let sample_port = self.ports.length()
      for index in sample_port..<self.pools.length() {
        let bytes = self.download_bytes(index)
        if bytes == 0 {
          continue
        }
        self.pools[index].read_host(
          self.payloads[index],
          host_offset=0L,
          destination_offset=0,
          bytes~,
        )
        self.pools[index].reset(0)
        if index > sample_port && load_i32(self.payloads[index], 0) != 0 {
          self.failed = true
        }
      }
      if self.outputs_token {
        for row in 0..<self.frame.output_rows() {
          let token = load_i32(self.payloads[sample_port], row)
          self.samples[row] = token
          if token < 0 || token >= self.vocabulary {
            self.failed = true
          }
        }
      }
      self.state = Retired
      true
    }
    Retired => true
    _ => raise @frame.BatchFrameError::Busy
  }
}

///|
pub fn BatchedDecoderRank::execution_failed(self : Self) -> Bool raise {
  guard self.state is Retired else { raise @frame.BatchFrameError::Busy }
  self.failed
}

///|
pub fn BatchedDecoderRank::samples(self : Self) -> ArrayView[Int] raise {
  guard self.state is Retired && self.outputs_token && !self.failed else {
    raise @frame.BatchFrameError::Busy
  }
  self.samples[:self.frame.output_rows()]
}

///|
pub fn BatchedDecoderRank::live_rows(self : Self) -> Int raise {
  self.frame.live_rows()
}

///|
pub fn BatchedDecoderRank::output_rows(self : Self) -> Int raise {
  if self.outputs_token {
    self.frame.output_rows()
  } else {
    0
  }
}

///|
/// Preserves the request-owned sampling configuration before GPU submission.
/// Binding a greedy-only suffix is not sufficient for stochastic requests.
pub fn BatchedDecoderRank::sampling_plan(
  self : Self,
) -> @wire.ValidatedPlanFrame raise {
  self.frame.sampling_plan()
}

///|
pub fn BatchedDecoderRank::output_entry(self : Self, output : Int) -> Int raise {
  self.frame.output_entry(output)
}

///|
pub fn BatchedDecoderRank::commit(
  self : Self,
  pipeline_failed~ : Bool,
) -> Unit raise {
  guard self.state is Retired else { raise @frame.BatchFrameError::Busy }
  self.state = Failed
  self.frame.retire_stage(execution_failed=pipeline_failed || self.failed)
  if !self.frame.is_faulted() {
    self.state = Ready
  }
}

///|
pub fn BatchedDecoderRank::release_request(
  self : Self,
  request_id : UInt64,
  generation : UInt64,
) -> Unit raise {
  guard self.state is Ready else { raise @frame.BatchFrameError::Busy }
  self.frame.release_request(request_id, generation)
}
