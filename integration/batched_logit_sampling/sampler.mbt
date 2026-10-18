///|
priv enum State {
  Ready
  Prepared
  Downloading
  Retired
  Failed
  Closed
}

///|
/// Canonical host sampler on literal row-major F32 model logits. A prepared
/// bounded DMA downloads only live output rows; no model data or comparison
/// is substituted. The same scalar sampler is shared with ordinary serving.
pub struct BatchedLogitSampler {
  priv pool : @device.TransferPool
  priv bytes : FixedArray[Byte]
  priv logits : Array[Double]
  priv limits : @sampling.SamplingLimits
  priv scratch : @sampling.SamplerScratch
  priv vocabulary : Int
  priv modes : FixedArray[Int]
  priv temperatures : FixedArray[Double]
  priv top_k : FixedArray[Int]
  priv top_p : FixedArray[Double]
  priv flags : FixedArray[Int]
  priv seeds : FixedArray[UInt64]
  priv indices : FixedArray[Int]
  priv tokens : FixedArray[Int]
  priv mut count : Int
  priv mut state : State
}

///|
pub fn BatchedLogitSampler::new(
  context : @device.Context,
  logits : @device.Allocation,
  vocabulary_size~ : Int,
  maximum_outputs~ : Int,
  maximum_host_bytes~ : Int64,
) -> Self raise {
  let limits = @sampling.SamplingLimits::new(maximum_outputs, vocabulary_size)
  let bytes = vocabulary_size.to_int64() * maximum_outputs.to_int64() * 4L
  // Conservative reservation: pinned and byte staging, row doubles and
  // canonical sampler scratch (integer IDs/F64 weights), and output metadata.
  let host = bytes * 2L +
    vocabulary_size.to_int64() * 32L +
    maximum_outputs.to_int64() * 64L
  guard vocabulary_size > 0 &&
    maximum_outputs > 0 &&
    bytes <= 2147483647L &&
    host <= maximum_host_bytes &&
    logits.byte_count() >= bytes else {
    raise @device.DeviceError::InvalidArgument(DeviceCopyCreate)
  }
  let scratch = @sampling.SamplerScratch::new(vocabulary_size, limits)
  let pool = context.prepare_transfer_pool(
    logits,
    host_bytes=bytes,
    lanes=1,
    max_regions_per_lane=1,
  )
  errdefer (pool.close() catch { _ => () })
  pool.check_prepared()
  {
    pool,
    bytes: FixedArray::make(bytes.to_int(), b'\x00'),
    logits: Array::make(vocabulary_size, 0.0),
    limits,
    scratch,
    vocabulary: vocabulary_size,
    modes: FixedArray::make(maximum_outputs, 0),
    temperatures: FixedArray::make(maximum_outputs, 1.0),
    top_k: FixedArray::make(maximum_outputs, 0),
    top_p: FixedArray::make(maximum_outputs, 0.0),
    flags: FixedArray::make(maximum_outputs, 0),
    seeds: FixedArray::make(maximum_outputs, 0UL),
    indices: FixedArray::make(maximum_outputs, 0),
    tokens: FixedArray::make(maximum_outputs, 0),
    count: 0,
    state: Ready,
  }
}

///|
fn BatchedLogitSampler::prepare(
  self : Self,
  plan : @wire.ValidatedPlanFrame,
) -> Unit raise {
  guard self.state is (Ready | Retired) else {
    raise @device.DeviceError::Busy(DeviceCopyCreate)
  }
  self.state = Failed
  self.count = 0
  for entry in 0..<(plan.prefill_count() + plan.decode_count()) {
    let prefill = entry < plan.prefill_count()
    if prefill && !plan.prefill_row(entry).produces_token() {
      continue
    }
    let i = self.count
    guard i < self.tokens.length() else {
      raise @device.DeviceError::InvalidArgument(DeviceCopyCreate)
    }
    if prefill {
      let row = plan.prefill_row(entry)
      self.modes[i] = row.sampling_mode_value()
      self.temperatures[i] = row.sampling_temperature()
      self.top_k[i] = row.sampling_top_k_value()
      self.top_p[i] = row.sampling_top_p_value()
      self.flags[i] = (if row.sampling_has_top_k() { 1 } else { 0 }) |
        (if row.sampling_has_top_p() { 2 } else { 0 })
      self.seeds[i] = row.sampling_seed_value()
      self.indices[i] = row.sample_index()
    } else {
      let row = plan.decode_row(entry - plan.prefill_count())
      self.modes[i] = row.sampling_mode_value()
      self.temperatures[i] = row.sampling_temperature()
      self.top_k[i] = row.sampling_top_k_value()
      self.top_p[i] = row.sampling_top_p_value()
      self.flags[i] = (if row.sampling_has_top_k() { 1 } else { 0 }) |
        (if row.sampling_has_top_p() { 2 } else { 0 })
      self.seeds[i] = row.sampling_seed_value()
      self.indices[i] = row.sample_index()
    }
    self.count += 1
  }
  self.state = Prepared
}

///|
fn BatchedLogitSampler::begin(self : Self) -> Unit raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::Busy(DeviceCopyCreate)
  }
  self.state = Failed
  if self.count > 0 {
    self.pool.enqueue(
      lane=0,
      host_offset=0L,
      device_offset=0L,
      bytes=self.count.to_int64() * self.vocabulary.to_int64() * 4L,
      to_device=false,
    )
    self.pool.record(0)
  }
  self.state = Downloading
}

///|
fn BatchedLogitSampler::poll(self : Self) -> Bool raise {
  if self.state is Retired {
    return true
  }
  guard self.state is Downloading else {
    raise @device.DeviceError::Busy(DeviceCopyCreate)
  }
  if self.count > 0 && !self.pool.poll(0) {
    return false
  }
  self.state = Failed
  if self.count > 0 {
    self.pool.read_host(
      self.bytes,
      host_offset=0L,
      destination_offset=0,
      bytes=self.count * self.vocabulary * 4,
    )
    self.pool.reset(0)
  }
  for row in 0..<self.count {
    for word in 0..<self.vocabulary {
      let offset = (row * self.vocabulary + word) * 4
      let bits = self.bytes[offset].to_uint() |
        (self.bytes[offset + 1].to_uint() << 8) |
        (self.bytes[offset + 2].to_uint() << 16) |
        (self.bytes[offset + 3].to_uint() << 24)
      self.logits[word] = bits.reinterpret_as_float().to_double()
    }
    self.tokens[row] = try {
      if self.modes[row] == 0 {
        @sampling.greedy(self.logits[:], self.limits).token_id()
      } else {
        @sampling.stochastic_sample_at_scalars(
          self.logits[:],
          temperature=self.temperatures[row],
          has_top_k=(self.flags[row] & 1) != 0,
          top_k=self.top_k[row],
          has_top_p=(self.flags[row] & 2) != 0,
          top_p=self.top_p[row],
          self.limits,
          @inference.SamplingSeed::new(self.seeds[row]),
          self.indices[row],
          self.scratch,
        ).token_id()
      }
    } catch {
      _ => -1
    }
  }
  self.state = Retired
  true
}

///|
pub fn BatchedLogitSampler::binding(self : Self) -> @rank.BatchedSampling raise {
  @rank.BatchedSampling::new(
    maximum_outputs=self.tokens.length(),
    vocabulary_size=self.vocabulary,
    prepare=plan => self.prepare(plan),
    begin=() => self.begin(),
    poll=() => self.poll(),
    token_at=row => {
      guard self.state is Retired && row >= 0 && row < self.count else {
        raise @device.DeviceError::Busy(DeviceCopyCreate)
      }
      self.tokens[row]
    },
    close=() => self.close(),
  )
}

///|
pub fn BatchedLogitSampler::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.pool.drain(0)
  self.pool.close()
  self.state = Closed
}
