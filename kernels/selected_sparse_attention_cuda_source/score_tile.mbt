///|
/// NVIDIA cooperative copy width is private. The portable tile owns the
/// swizzled storage and the copy -> publish -> ordered fold -> retire phases.
fn score_tile_source(
  schedule : @physical.SelectedSparseSchedule,
  dimension : Int,
  kv_heads : Int,
  scalar : String,
  branch_storage? : String,
) -> String raise {
  guard dimension >= 32 && schedule.score_tile(32) is Some(tile) else {
    return scalar
  }
  let (keys, components) = tile.geometry()
  let dot = expression(DotUpdate, [
    "dot",
    "tile_query[component]",
    "tile_keys[component*\{keys}+(worker^(component*2))]",
  ])
  let score = expression(Score, ["dot", "scale"])
  let storage = match branch_storage {
    None =>
      "__shared__ __nv_bfloat16 tile_keys[\{keys*components}],tile_query[\{components}];"
    Some(pointer) =>
      "__nv_bfloat16 *tile_keys=(__nv_bfloat16*)\{pointer},*tile_query=tile_keys+\{keys*components};"
  }
  (
    $|if(visible>=\{keys}) {
    $| // Cooperative key rows, swizzled copy ownership, ordered score folds.
    $| \{storage}
    $| for(int first=0;first<visible;first+=\{keys}) {
    $|  float dot=0.0f;
    $|  const int item=first+worker;
    $|  for(int begin=0;begin<\{dimension};begin+=\{components}) {
    $|   const int width=min(\{components},\{dimension}-begin);
    $|   for(int cell=worker;cell<\{keys*components};cell+=blockDim.x) {
    $|    const int key=cell/\{components},component=cell%\{components};
    $|    __nv_bfloat16 value=__float2bfloat16_rn(0.0f);
    $|    if(first+key<visible&&component<width) {
    $|     const unsigned long long base=((unsigned long long)retained[first+key]*\{kv_heads}+kv_head)*\{dimension}ULL;
    $|     value=keys[base+begin+component];
    $|    }
    $|    tile_keys[component*\{keys}+(key^(component*2))]=value;
    $|   }
    $|   if(worker<width)tile_query[worker]=queries[qbase+begin+worker];
    $|   __syncthreads(); // cooperative copy publishes before all score readers
    $|   if(item<visible)for(int component=0;component<width;++component)dot=\{dot};
    $|   __syncthreads(); // readers retire before the next component window
    $|  }
    $|  if(item<visible)scores[item]=\{score};
    $| }
    $|} else {
    $| \{scalar}
    $|}
  )
}
