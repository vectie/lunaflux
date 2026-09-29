///|
/// A conservative immutable partition of the value graph. Until alias
/// independence is proved, every observable write ends a scheduling region.
/// The terminating write belongs to the preceding region so that an exclusive
/// producer chain can fuse into its own store/commit, but never across it.
priv struct ProjectionEffectRegionEntry {
  value : ProjectionTileValueId
  region : Int
}

///|
fn projection_effect_regions(
  program : ProjectionTileProgram,
) -> ReadOnlyArray[ProjectionEffectRegionEntry] {
  let entries : Array[ProjectionEffectRegionEntry] = []
  for binding in program.bindings; region = 0 {
    entries.push({ value: binding.id, region, })
    continue region +
      (if projection_has_effect(binding.expression) { 1 } else { 0 })
  }
  ReadOnlyArray::from_array(entries)
}

///|
fn projection_same_effect_region(
  regions : ReadOnlyArray[ProjectionEffectRegionEntry],
  values : ArrayView[ProjectionTileValueId],
) -> Bool {
  for value in values; expected = (None : Int?) {
    let found = for entry in regions {
      if entry.value == value {
        break Some(entry.region)
      }
    } nobreak {
      None
    }
    guard found is Some(region) else { return false }
    if expected is Some(prior) && prior != region {
      return false
    }
    continue Some(region)
  } nobreak {
    expected is Some(_)
  }
}
