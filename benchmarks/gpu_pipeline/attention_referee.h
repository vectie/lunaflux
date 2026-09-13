// Offline independent scalar oracle. No production dependency.
static float read_bf16(const std::vector<unsigned char>& bytes, size_t index) {
  __nv_bfloat16 value;
  std::memcpy(&value, bytes.data() + index * 2, 2);
  return __bfloat162float(value);
}
static void check_attention_referee(
    const std::vector<unsigned char>& input,
    const std::vector<unsigned char>& keys,
    const std::vector<unsigned char>& values,
    const std::vector<unsigned char>& output,
    const std::vector<int>& positions, const std::vector<int>& offsets,
    const std::vector<int>& page_offsets, const std::vector<int>& pages) {
  float maximum_error = 0;
  for (size_t row = 0; row + 1 < offsets.size(); ++row) {
    for (int token : {offsets[row], (offsets[row] + offsets[row+1] - 1)/2,
                      offsets[row+1] - 1}) {
      for (int head : {0, 15}) {
        const int context = positions[token] + 1;
        std::vector<float> scores(context);
        float maximum = -INFINITY;
        for (int key = 0; key < context; ++key) {
          const size_t address = size_t(pages[page_offsets[row]+key/8])*8192 +
                                (key%8*8+head/2)*128;
          float sum = 0;
          for (int d=0; d<128; ++d)
            sum += read_bf16(input,size_t(token)*4096+head*128+d)*
                   read_bf16(keys,address+d);
          scores[key] = sum / std::sqrt(128.0f);
          maximum = std::fmax(maximum,scores[key]);
        }
        double denominator = 0;
        for (float& score : scores) { score = std::exp(score-maximum); denominator += score; }
        for (int d=0; d<128; ++d) {
          double sum=0;
          for (int key=0; key<context; ++key) {
            const size_t address=size_t(pages[page_offsets[row]+key/8])*8192+(key%8*8+head/2)*128+d;
            sum += scores[key]*read_bf16(values,address);
          }
          const float actual=read_bf16(output,size_t(token)*2048+head*128+d);
          const float error=std::fabs(actual-float(sum/denominator));
          if (!std::isfinite(actual) || error>0.003f) {
            std::fprintf(stderr,"independent referee mismatch token=%d head=%d d=%d abs=%g\n",token,head,d,error);
            std::exit(5);
          }
          maximum_error=std::fmax(maximum_error,error);
        }
      }
    }
  }
  std::printf("independent_scalar_referee_maxabs=%g\n",maximum_error);
}
