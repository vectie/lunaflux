///|
/// Physical placement of an existing descending pairwise tree. Moving its
/// final levels into a subgroup must not regroup the leaves: a subgroup-first
/// reduction would be a different floating-point expression.
pub struct LunaTileOrderedReduction {
  priv shared_strides : ReadOnlyArray[Int]
  priv subgroup_strides : ReadOnlyArray[Int]
} derive(Debug, Eq)

///|
/// Plans storage/communication only; every addition retains its original two
/// operands and order. Geometry is supplied by a backend, not model identity.
pub fn LunaTileOrderedReduction::new(
  lanes~ : Int,
  subgroup_lanes~ : Int,
) -> LunaTileOrderedReduction raise LunaTileError {
  if lanes <= 0 ||
    lanes > 1024 ||
    (lanes & (lanes - 1)) != 0 ||
    subgroup_lanes <= 0 ||
    subgroup_lanes > lanes ||
    (subgroup_lanes & (subgroup_lanes - 1)) != 0 {
    raise InvalidProgram(Reduction)
  }
  let shared : Array[Int] = []
  let subgroup : Array[Int] = []
  for stride = lanes / 2; stride > 0; stride = stride / 2 {
    if stride >= subgroup_lanes {
      shared.push(stride)
    } else {
      subgroup.push(stride)
    }
  }
  {
    shared_strides: ReadOnlyArray::from_array(shared),
    subgroup_strides: ReadOnlyArray::from_array(subgroup),
  }
}

///|
pub fn LunaTileOrderedReduction::shared_strides(
  self : LunaTileOrderedReduction,
) -> ReadOnlyArray[Int] {
  self.shared_strides
}

///|
pub fn LunaTileOrderedReduction::subgroup_strides(
  self : LunaTileOrderedReduction,
) -> ReadOnlyArray[Int] {
  self.subgroup_strides
}

///|
/// A rounded producer may be forwarded across the fold only under an explicit
/// per-lane register budget. None preserves the materialized-output route.
pub fn luna_tile_retained_row_values(
  elements~ : Int,
  lanes~ : Int,
  maximum_values_per_lane~ : Int,
) -> Int? {
  if elements <= 0 || lanes <= 0 || maximum_values_per_lane <= 0 {
    return None
  }
  let values = (elements - 1) / lanes + 1
  if values <= maximum_values_per_lane {
    Some(values)
  } else {
    None
  }
}
