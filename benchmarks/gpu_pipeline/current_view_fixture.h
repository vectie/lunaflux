// The producer contract commits the same positioned K/V into the cache and
// current input view. Physical pages are independently permuted by the probe.
static void initialize_current_view(Buffer& input, Buffer& keys, Buffer& values,
    const std::vector<int>& positions, const std::vector<int>& offsets,
    const std::vector<int>& page_offsets, const std::vector<int>& pages) {
  auto data=input.read(), k=keys.read(), v=values.read();
  for (size_t row=0;row+1<offsets.size();++row) {
    for (int token=offsets[row];token<offsets[row+1];++token) {
      const int p=positions[token];
      const size_t address=size_t(pages[page_offsets[row]+p/8])*8192+(p%8)*1024;
      std::memcpy(data.data()+size_t(token)*8192+4096,k.data()+address*2,2048);
      std::memcpy(data.data()+size_t(token)*8192+6144,v.data()+address*2,2048);
    }
  }
  CK(cudaMemcpy(input.p,data.data(),data.size(),cudaMemcpyHostToDevice));
}
