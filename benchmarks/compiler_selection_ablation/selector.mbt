///|
fn main raise {
  let args = @env.args()
  let (args, samples) = match args[:] {
    [.. rest, "--samples", count] => (rest, @string.parse_int(count))
    _ => (args[:], 5)
  }
  guard args.length() >= 7 && (args.length() - 3) % 4 == 0 else {
    fail("MAX_LOCAL MAX_WORKSPACE [ID NS LOCAL WORKSPACE]... [--samples COUNT]")
  }
  let graph = @fusion_regions.Graph::new([
    @fusion_regions.Node::new([], observable=true, effect=Pure),
  ])
  let partition = @fusion_regions.Partition::new(graph, [
    @fusion_regions.Region::new(graph, start=0, count=1, [0]),
  ])
  let alternatives = [
    for index in 0..<((args.length() - 3) / 4) => {
      let offset = 3 + index * 4
      @fusion_regions.PartitionAlternative::new(
        id=@string.parse_int(args[offset]),
        partition,
        @fusion_regions.Cost::measured(
          nanoseconds=@string.parse_int64(args[offset + 1]),
          samples~,
        ),
        local_bytes=@string.parse_int64(args[offset + 2]),
        workspace_bytes=@string.parse_int64(args[offset + 3]),
      )
    }
  ]
  let id = @fusion_regions.select_partition(
    alternatives,
    maximum_local_bytes=@string.parse_int64(args[1]),
    maximum_workspace_bytes=@string.parse_int64(args[2]),
  )
  println(id)
}
