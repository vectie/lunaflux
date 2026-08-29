///|
fn close_optional(
  resource : @device.Allocation?,
  stage : DeviceStepStage,
) -> Unit raise DeviceStepError {
  match resource {
    None => ()
    Some(allocation) =>
      allocation.close() catch {
        error => raise Device(stage, error)
      }
  }
}

///|
fn close_partial(
  resources : PartialDeviceStepResources,
) -> Unit raise DeviceStepError {
  let mut failure : DeviceStepError? = None
  let close = (resource, stage) => {
    try close_optional(resource, stage) catch {
      error => if failure is None { failure = Some(error) }
    } noraise {
      _ => ()
    }
  }
  close(resources.physical_page_indices, PhysicalPageIndices)
  close(resources.page_offsets, PageOffsets)
  close(resources.sequence_lengths, SequenceLengths)
  close(resources.query_row_offsets, QueryRowOffsets)
  close(resources.token_positions, TokenPositions)
  close(resources.token_ids, TokenIds)
  close(resources.counts, Counts)
  match failure {
    Some(error) => raise error
    None => ()
  }
}

///|
fn partial_from_complete(
  resources : DeviceStepResources,
) -> PartialDeviceStepResources {
  {
    counts: Some(resources.counts),
    token_ids: Some(resources.token_ids),
    token_positions: Some(resources.token_positions),
    query_row_offsets: Some(resources.query_row_offsets),
    sequence_lengths: Some(resources.sequence_lengths),
    page_offsets: Some(resources.page_offsets),
    physical_page_indices: Some(resources.physical_page_indices),
  }
}

///|
fn close_with_lifecycle(
  lifecycle : DeviceStepLifecycle,
  set_lifecycle : (DeviceStepLifecycle) -> Unit,
  cleanup : () -> Unit raise DeviceStepError,
) -> Unit raise DeviceStepError {
  if lifecycle == Closed {
    return
  }
  set_lifecycle(ClosingFailed)
  cleanup()
  set_lifecycle(Closed)
}

///|
pub fn DeviceStepOwner::close(
  self : DeviceStepOwner,
) -> Unit raise DeviceStepError {
  close_with_lifecycle(
    self.lifecycle,
    lifecycle => self.lifecycle = lifecycle,
    () => close_partial(partial_from_complete(self.resources)),
  )
}

///|
pub fn FailedDeviceStepPreparation::retry_close(
  self : FailedDeviceStepPreparation,
) -> Unit raise DeviceStepError {
  close_with_lifecycle(
    self.lifecycle,
    lifecycle => self.lifecycle = lifecycle,
    () => {
      let outcome : Result[Unit, DeviceStepError] = Ok(
        close_partial(self.resources),
      ) catch {
        error => Err(error)
      }
      match outcome {
        Err(error) => {
          self.cleanup_failure = error
          raise error
        }
        Ok(_) => ()
      }
    },
  )
}
