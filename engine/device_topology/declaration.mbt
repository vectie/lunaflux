///|
pub fn LocalTopologyRank::new(
  ordinal~ : Int,
  name~ : StringView,
  total_memory_bytes~ : Int64,
  target~ : @catalog.DeviceTarget,
) -> LocalTopologyRank raise LocalTopologyError {
  if ordinal < 0 || ordinal > 1023 {
    raise invalid(DeclarationRanks, InvalidOrdinal, 0, 0)
  }
  if name.is_empty() || name.length() > MAX_DEVICE_NAME_UNITS {
    raise invalid(DeclarationRanks, InvalidName, 0, 0)
  }
  if total_memory_bytes <= 0L {
    raise invalid(DeclarationRanks, InvalidMemory, 0, 0)
  }
  { ordinal, name: name.to_owned(), total_memory_bytes, target }
}

///|
pub fn LocalTopologyLink::direct(
  lower_rank~ : Int,
  upper_rank~ : Int,
) -> LocalTopologyLink raise LocalTopologyError {
  if lower_rank < 0 || upper_rank < 0 || lower_rank >= upper_rank {
    raise invalid(DeclarationLinks, Unsupported, lower_rank, upper_rank)
  }
  { lower_rank, upper_rank }
}

///|
fn expected_link_count(rank_count : Int) -> Int {
  rank_count * (rank_count - 1) / 2
}

///|
fn validate_declaration_ranks(
  ranks : ArrayView[LocalTopologyRank],
) -> Unit raise LocalTopologyError {
  if ranks.length() < MIN_LOCAL_RANKS {
    raise invalid(DeclarationRanks, TooFew, ranks.length(), MIN_LOCAL_RANKS)
  }
  if ranks.length() > MAX_LOCAL_RANKS {
    raise invalid(DeclarationRanks, TooMany, ranks.length(), MAX_LOCAL_RANKS)
  }
  let expected_target = ranks[0].target
  for index, rank in ranks {
    if index > 0 && rank.ordinal <= ranks[index - 1].ordinal {
      raise invalid(
        DeclarationRanks,
        if rank.ordinal == ranks[index - 1].ordinal {
          Duplicate
        } else {
          RankOrder
        },
        index - 1,
        index,
      )
    }
    if rank.target != expected_target {
      raise invalid(DeclarationRanks, HeterogeneousTarget, 0, index)
    }
  }
}

///|
fn validate_declaration_links(
  rank_count : Int,
  links : ArrayView[LocalTopologyLink],
) -> Unit raise LocalTopologyError {
  let required = expected_link_count(rank_count)
  if links.length() != required {
    raise invalid(DeclarationLinks, Incomplete, links.length(), required)
  }
  let mut cursor = 0
  for lower in 0..<rank_count {
    for upper in (lower + 1)..<rank_count {
      let link = links[cursor]
      if link.lower_rank != lower || link.upper_rank != upper {
        raise invalid(DeclarationLinks, Unsupported, lower, upper)
      }
      cursor += 1
    }
  }
}

///|
/// Admits only the initial Phase-7 local homogeneous full mesh. This is a
/// LunaFlux support boundary, not a claim about general NCCL requirements.
pub fn LocalTopologyDeclaration::new(
  ranks : ArrayView[LocalTopologyRank],
  links : ArrayView[LocalTopologyLink],
) -> LocalTopologyDeclaration raise LocalTopologyError {
  validate_declaration_ranks(ranks)
  validate_declaration_links(ranks.length(), links)
  {
    ranks: ReadOnlyArray::from_array(ranks),
    links: ReadOnlyArray::from_array(links),
  }
}
