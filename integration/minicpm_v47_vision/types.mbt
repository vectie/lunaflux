///|
pub(all) enum Plane {
  RawCounts
  EncodedCounts
  OutputCounts
  Pixels
  PositionIds
  RawStarts
  RawEnds
  EncodedStarts
  EncodedEnds
  WindowStarts
  WindowEnds
  WindowIndices
  MergerIndices
  State
  Normalized
  Query
  Key
  Value
  Attention
  Branch
  Intermediate
  Features
} derive(Eq)

///|
pub extend Plane with Eq::{equal, not_equal}

///|
/// Buffer planes are typed, while checkpoint tensor names resolve once at
/// startup. No strings or dictionary lookups belong in frame submission.
pub(all) enum Argument {
  Buffer(Plane)
  Weight(String)
}

///|
pub struct Entry {
  priv symbol : String
  priv launch : (Int32, Int32, Int32)
  priv parameters : Int32
  priv shared_bytes : Int32
}

///|
pub fn Entry::symbol(self : Self) -> String {
  self.symbol
}

///|
pub fn Entry::launch(self : Self) -> (Int32, Int32, Int32) {
  self.launch
}

///|
pub fn Entry::parameters(self : Self) -> Int32 {
  self.parameters
}

///|
/// Statically declared shared memory, not a dynamic launch allocation.
pub fn Entry::shared_bytes(self : Self) -> Int32 {
  self.shared_bytes
}

///|
pub struct Stage {
  priv entry : Int32
  priv arguments : ReadOnlyArray[Argument]
}

///|
pub fn Stage::entry(self : Self) -> Int32 {
  self.entry
}

///|
pub fn Stage::arguments(self : Self) -> ReadOnlyArray[Argument] {
  self.arguments
}

///|
pub(all) struct Region {
  plane : Plane
  offset : Int64
  bytes : Int64
}

///|
/// One immutable AOT module and the full vision launch sequence. Device
/// loading, native frame binding and media/request lifetime remain separate.
pub struct Program {
  priv source : Bytes
  priv entries : ReadOnlyArray[Entry]
  priv stages : ReadOnlyArray[Stage]
  priv regions : ReadOnlyArray[Region]
  priv workspace_bytes : Int64
  priv rows : (Int32, Int32, Int32)
  priv crop_rows : Int32
}

///|
pub fn Program::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn Program::entries(self : Self) -> ReadOnlyArray[Entry] {
  self.entries
}

///|
pub fn Program::stages(self : Self) -> ReadOnlyArray[Stage] {
  self.stages
}

///|
pub fn Program::regions(self : Self) -> ReadOnlyArray[Region] {
  self.regions
}

///|
pub fn Program::workspace_bytes(self : Self) -> Int64 {
  self.workspace_bytes
}

///|
/// Capacity before early merge, after encoder, and after final merger.
pub fn Program::row_capacities(self : Self) -> (Int32, Int32, Int32) {
  self.rows
}

///|
pub fn Program::maximum_crop_patches(self : Self) -> Int32 {
  self.crop_rows
}

///|
pub fn Program::weight_names(self : Self) -> ReadOnlyArray[String] {
  let names : Map[String, Bool] = Map([])
  for stage in self.stages {
    for argument in stage.arguments {
      if argument is Weight(name) {
        names[name] = true
      }
    }
  }
  ReadOnlyArray::from_array(names.keys().collect())
}
