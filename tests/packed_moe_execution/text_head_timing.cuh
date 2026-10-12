// Test-only CUDA-event timing. Never linked into a runtime.
template<class Launch> static void time_text_head(Launch launch) {
  cudaEvent_t start, end;
  check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
  for(int i=0;i<50;++i) launch();
  check(cudaDeviceSynchronize());
  for(int sample=0;sample<7;++sample) {
    check(cudaEventRecord(start));
    for(int i=0;i<500;++i) launch();
    check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
    float ms=0; check(cudaEventElapsedTime(&ms,start,end));
    std::printf("head_fixture_sample=%d suffix_us=%.6f iterations=500\n",sample,ms*1000.f/500.f);
  }
  check(cudaEventDestroy(end)); check(cudaEventDestroy(start));
}
