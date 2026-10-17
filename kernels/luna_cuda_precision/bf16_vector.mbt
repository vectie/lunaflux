///|
/// Physical eight-value transport of the existing decoded-BF16 operand.
/// Finite values (including signed zero) retain their bits. The scalar loader's
/// max(-finite, value), min(finite, value) maps NaN to -finite and infinity to
/// signed finite. Integer selection preserves that law without float roundtrips.
fn bf16_tile_vector(row_bytes : Int64, symbol~ : String) -> String {
  let words = ["x", "y", "z", "w"]
  let body = StringBuilder()
  for word in words {
    body <+
      $|  {unsigned a=bits.\{word}&65535U,b=bits.\{word}>>16;
      $|   a=(a&32767U)>32640U?65407U:(a&32767U)==32640U?(a&32768U)|32639U:a;
      $|   b=(b&32767U)>32640U?65407U:(b&32767U)==32640U?(b&32768U)|32639U:b;
      $|   bits.\{word}=a|(b<<16);}
      $|
  }
  prelude() +
  (
    $|__device__ __forceinline__ uint4 \{symbol}(const unsigned char *payload,const unsigned char *scales,const float *global,unsigned long long row,unsigned long long column) {
    $|  const unsigned char *p=payload+row*\{row_bytes}ULL+column*2ULL;
    $|  uint4 bits;
    $|  if((((unsigned long long)p)&15ULL)==0ULL) bits=*reinterpret_cast<const uint4 *>(p);
    $|  else bits=make_uint4(lf_precision_read(p,0,4),lf_precision_read(p,4,4),lf_precision_read(p,8,4),lf_precision_read(p,12,4));
    $|  \{body.to_string()}
    $|  return bits;
    $|}
  )
}
