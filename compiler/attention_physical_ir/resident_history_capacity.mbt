///|
/// Startup capacity for full retained history, independent of request slots.
/// The page pool is shared, not maximum_context * maximum_sequences. Reserve
/// pages remain available for decode while prefill competes for the pool.
/// This is pure geometry; allocation, admission and eviction are runtime effects.
pub struct ResidentHistoryCapacity {
  priv domain : PagedHistoryDomain
  priv reserve : Int
}

///|
pub fn ResidentHistoryCapacity::new(
  rows_per_page~ : Int,
  physical_pages~ : Int,
  maximum_context~ : Int,
  maximum_sequences~ : Int,
  decode_reserve_pages~ : Int,
) -> Self? {
  guard decode_reserve_pages >= 0 && decode_reserve_pages < physical_pages else {
    return None
  }
  guard PagedHistoryDomain::new(
      rows_per_page~,
      physical_pages~,
      maximum_context~,
      maximum_sequences~,
    )
    is Some(domain) else {
    return None
  }
  // A full-history service must be able to finish even one admitted maximum
  // prompt without relying on eviction/recompute to create nonexistent space.
  guard domain.pages_for(maximum_context).unwrap() <=
    physical_pages - decode_reserve_pages else {
    return None
  }
  Some({ domain, reserve: decode_reserve_pages, })
}

///|
pub fn ResidentHistoryCapacity::domain(self : Self) -> PagedHistoryDomain {
  self.domain
}

///|
pub fn ResidentHistoryCapacity::decode_reserve_pages(self : Self) -> Int {
  self.reserve
}

///|
/// Conservative number of simultaneous requests at this declared length.
/// Shorter requests may occupy more slots; queued requests own no history.
pub fn ResidentHistoryCapacity::resident_sequences_at(
  self : Self,
  tokens : Int,
) -> Int? {
  guard tokens > 0 && self.domain.pages_for(tokens) is Some(pages) else {
    return None
  }
  let (_, physical, _, slots) = self.domain.geometry()
  Some(Int::min(slots, (physical - self.reserve) / pages))
}
