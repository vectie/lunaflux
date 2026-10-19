///|
fn uniqueness_source(
  schedule : @physical.SelectedSparseSchedule,
  extent : Int,
) -> (String, String) {
  match schedule.uniqueness(extent).unwrap() {
    Pairwise =>
      (
        (
          $|for (int prior = 0; prior < rank; ++prior)
          $|  if (selected_indices[selected_base + prior] == key) { atomicExch(&row_valid, 0); break; }
        ),
        "",
      )
    WindowedBitmap(words~) =>
      (
        "",
        (
          $|// Exact bounded membership: no quadratic per-head duplicate scan.
          $|__shared__ unsigned int seen[\{words}];
          $|for(int first=0;first<=position;first+=\{words*32}) {
          $| for(int word=worker;word<\{words};word+=blockDim.x)seen[word]=0;
          $| __syncthreads();
          $| for(int rank=worker;rank < \{extent};rank+=blockDim.x) {
          $|  const int key=selected_indices[selected_base+rank],relative=key-first;
          $|  if(relative>=0&&relative<\{words*32}) {
          $|   const unsigned int bit=1U<<(relative%32);
          $|   if(atomicOr(&seen[relative/32],bit)&bit)atomicExch(&row_valid,0);
          $|  }
          $| }
          $| __syncthreads(); // membership readers retire before window reuse
          $| if(!row_valid)return;
          $|}
        ),
      )
  }
}
