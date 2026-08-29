///|
pub fn LunaRequestPreparationPool::try_begin_luna_framed_with_receipt(
  self : LunaRequestPreparationPool,
  receipt : LunaRequestReceipt,
  source : FixedArray[Byte],
  source_offset~ : Int,
  length~ : Int,
) -> LunaRequestPreparationAdmission raise RequestAdmissionError {
  // Exact receipt authentication precedes pool disposition and source rules.
  let receipt_workspace = receipt.require_workspace()
  if self.draining {
    return preparation_admission(
      self,
      LunaRequestPreparationDraining,
      -1,
      0UL,
      0,
    )
  }
  if self.free_count == 0 {
    return preparation_admission(
      self,
      LunaRequestPreparationSaturated,
      -1,
      0UL,
      0,
    )
  }
  let next_lane = self.free_ring[self.free_head]
  if self.lanes[next_lane].generation == 0xffffffffffffffffUL {
    self.draining = true
    return preparation_admission(
      self,
      LunaRequestPreparationDraining,
      -1,
      0UL,
      0,
    )
  }
  if length <= 0 ||
    source_offset < 0 ||
    source_offset > source.length() - length {
    raise Invalid(FrameContract)
  }
  if receipt_workspace.budget_millis == 0UL ||
    receipt_workspace.budget_millis > self.inference.max_deadline_millis() {
    raise Invalid(Deadline)
  }
  // Range and saturation rejection above leave the Receipt live and avoid a
  // clock sample. Expiry and clock failure also leave it explicitly abortable.
  let observed = receipt.observe()
  let scanner = self.lanes[next_lane].framed_workspace.begin() catch {
    _ => raise Invalid(LunaPreparationPool)
  }
  let received_at_millis = receipt_workspace.received_at_millis
  let deadline_at_millis = receipt_workspace.deadline_at_millis
  let budget_millis = receipt_workspace.budget_millis
  let lane_index = self.pop_free_lane()
  let lane = self.lanes[lane_index]
  receipt.consume() catch {
    _ => abort("Luna trusted receipt changed during atomic transfer")
  }
  lane.generation += 1UL
  lane.receipt_at_millis = received_at_millis
  lane.hard_deadline = Some(
    @inference.AdmissionDeadline::from_receipt(
      received_at_millis,
      @inference.DeadlineBudget::from_milliseconds(
        budget_millis,
        self.inference,
      ) catch {
        _ => abort("Luna trusted receipt budget invariant")
      },
    ) catch {
      _ => abort("Luna trusted receipt deadline invariant")
    },
  )
  if lane.hard_deadline.unwrap().at_millis() != deadline_at_millis {
    abort("Luna trusted receipt absolute-deadline invariant")
  }
  lane.deadline = None
  lane.request = None
  lane.effective = None
  lane.framed_direct = true
  lane.framed_receipt_complete = false
  lane.trusted_receipt_bound = true
  lane.trusted_deadline_millis = budget_millis
  lane.framed_work.push(scanner)
  lane.phase = LUNA_PREPARATION_FRAMED_RECEIVING
  lane.failure = None
  lane.claim = None
  lane.last_work_units = 0
  lane.total_work_units = 0UL
  let offer_result : Result[Int, RequestAdmissionError] = Ok(
    self.offer_framed_lane(lane, source, source_offset, length, observed),
  ) catch {
    error => Err(error)
  }
  let consumed = match offer_result {
    Ok(value) => value
    Err(error) => {
      if lane.phase != LUNA_PREPARATION_FAILED {
        self.abort_lane_resources(lane)
      }
      self.reset_lane_to_idle(lane_index)
      raise error
    }
  }
  self.enqueue_active(lane_index)
  preparation_admission(
    self,
    LunaRequestPreparationAdmitted,
    lane_index,
    lane.generation,
    consumed,
  )
}
