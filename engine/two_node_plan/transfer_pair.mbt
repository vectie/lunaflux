///|
/// The initial plain-BF16 transfer vocabulary is the existing contiguous or
/// strided row copy. Both ranks must address the same authenticated tensor.
fn validate_transfer_pair(
  left : @tensor_parallel_plan.TensorRankPlacement,
  right : @tensor_parallel_plan.TensorRankPlacement,
) -> Unit raise TwoNodePlanError {
  for tensor in [left, right] {
    let recipe = tensor.transfer_recipe()
    let extent = tensor.extent()
    let destination = recipe.destination_offset_bytes_at(0L) catch {
      _ => raise Rejected(rule=TensorPartition, rank=-1)
    }
    if destination != tensor.region().offset_bytes() {
      raise Rejected(rule=TensorPartition, rank=-1)
    }
    match tensor.placement() {
      Replicated | ShardRows =>
        if recipe.segment_count() != 1L ||
          recipe.segment_byte_length() != extent.byte_length() {
          raise Rejected(rule=TensorPartition, rank=-1)
        }
      ShardColumns => {
        let row_bytes = checked_multiply(extent.column_count(), 2L)
        if recipe.segment_count() != extent.row_count() ||
          recipe.segment_byte_length() != row_bytes {
          raise Rejected(rule=TensorPartition, rank=-1)
        }
        // A recipe is affine; checking the last segment proves its stride
        // without a startup scan over potentially billions of tensor rows.
        let last = extent.row_count() - 1L
        if last > 0L {
          let first_source = recipe.source_payload_offset_bytes_at(0L) catch {
            _ => raise Rejected(rule=TensorPartition, rank=-1)
          }
          let last_source = recipe.source_payload_offset_bytes_at(last) catch {
            _ => raise Rejected(rule=TensorPartition, rank=-1)
          }
          let last_destination = recipe.destination_offset_bytes_at(last) catch {
            _ => raise Rejected(rule=TensorPartition, rank=-1)
          }
          if last_source - first_source !=
            checked_multiply(last, checked_multiply(extent.full_columns(), 2L)) ||
            last_destination - tensor.region().offset_bytes() !=
            checked_multiply(last, row_bytes) {
            raise Rejected(rule=TensorPartition, rank=-1)
          }
        }
      }
    }
  }
  let first = left.transfer_recipe().source_payload_offset_bytes_at(0L) catch {
      _ => raise Rejected(rule=TensorPartition, rank=-1)
    }
  let second = right.transfer_recipe().source_payload_offset_bytes_at(0L) catch {
      _ => raise Rejected(rule=TensorPartition, rank=-1)
    }
  let expected_delta = match left.placement() {
    Replicated => 0L
    ShardRows => left.extent().byte_length()
    ShardColumns => checked_multiply(left.extent().column_count(), 2L)
  }
  if second < first || second - first != expected_delta {
    raise Rejected(rule=TensorPartition, rank=-1)
  }
}
