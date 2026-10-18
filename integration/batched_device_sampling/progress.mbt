///|
pub fn BatchedDeviceSampler::begin(self : Self) -> Unit raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::Busy(OrderedExecutorEnqueue)
  }
  self.state = AwaitingInput
}

///|
pub fn BatchedDeviceSampler::poll(self : Self) -> Bool raise {
  match self.state {
    AwaitingInput => {
      self.state = Failed
      if !self.upload.poll(0) {
        self.state = AwaitingInput
        return false
      }
      self.upload.reset(0)
      if self.count == 0 {
        self.state = Retired
        return true
      }
      self.executor.launch_captured()
      self.executor.record_completion()
      self.state = Executing
      false
    }
    Executing => {
      self.state = Failed
      if !self.executor.poll_completion() {
        self.state = Executing
        return false
      }
      self.executor.reset()
      self.download.enqueue(
        lane=0,
        host_offset=0L,
        device_offset=0L,
        bytes=self.count.to_int64() * 8L,
        to_device=false,
      )
      self.download.record(0)
      self.state = Downloading
      false
    }
    Downloading => {
      self.state = Failed
      if !self.download.poll(0) {
        self.state = Downloading
        return false
      }
      self.download.read_host(
        self.output,
        host_offset=0L,
        destination_offset=0,
        bytes=self.count * 8,
      )
      self.download.reset(0)
      self.state = Retired
      true
    }
    Retired => true
    _ => raise @device.DeviceError::Busy(OrderedExecutorPoll)
  }
}

///|
pub fn BatchedDeviceSampler::binding(
  self : Self,
) -> @rank.BatchedSampling raise {
  let (vocabulary, _, rows) = self.program.geometry()
  @rank.BatchedSampling::new(
    maximum_outputs=rows,
    vocabulary_size=vocabulary,
    prepare=plan => self.prepare(plan),
    begin=() => self.begin(),
    poll=() => self.poll(),
    token_at=row => self.token_at(row),
    close=() => self.close(),
  )
}

///|
pub fn BatchedDeviceSampler::token_at(self : Self, row : Int32) -> Int32 raise {
  guard self.state is Retired && row >= 0 && row < self.count else {
    raise @device.DeviceError::Busy(DeviceCopyCreate)
  }
  if get_i32(self.output, row * 8 + 4) == -1 {
    get_i32(self.output, row * 8)
  } else {
    -1
  }
}

///|
pub fn BatchedDeviceSampler::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.upload.drain(0)
  self.executor.abort()
  self.download.drain(0)
  self.executor.close()
  self.download.close()
  self.upload.close()
  for f in self.functions.rev_iter() {
    f.close()
  }
  self.stream.close()
  self.results.close()
  self.stages.close()
  self.workspace.close()
  self.control.close()
  self.state = Closed
}
