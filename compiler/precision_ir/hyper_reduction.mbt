///|
/// A coherent numerical choice for control RMS and projection. The tree is
/// explicit semantics, never an implicit reassociation by device lowering.
pub(all) enum HyperControlReduction {
  Ordered
  Tree(partitions~ : Int)
}

///|
pub fn HyperControlReduction::square_law(self : Self) -> SquareReductionLaw {
  match self {
    Ordered => OrderedFloat32
    Tree(partitions~) => VectorizedTreeFloat32(partitions~, vector_width=1)
  }
}

///|
pub fn HyperControlReduction::dot_law(self : Self) -> DotReductionLaw {
  match self {
    Ordered => OrderedFloat32
    Tree(partitions~) => StripedTreeFloat32(partitions~)
  }
}
