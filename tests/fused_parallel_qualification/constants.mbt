///|
const FUSED_QUALIFICATION_BLOCK_THREADS : Int32 = 128

///|
const FUSED_QUALIFICATION_INPUT_WIDTH : Int32 = 128

///|
const FUSED_QUALIFICATION_QUERY_HEADS : Int32 = 4

///|
const FUSED_QUALIFICATION_KV_HEADS : Int32 = 2

///|
const FUSED_QUALIFICATION_HEAD_DIMENSION : Int32 = 32

///|
const FUSED_QUALIFICATION_QUERY_WIDTH : Int32 = 128

///|
const FUSED_QUALIFICATION_KV_WIDTH : Int32 = 64

///|
const FUSED_QUALIFICATION_OUTPUT_WIDTH : Int32 = 256

///|
const FUSED_QUALIFICATION_TOKENS_PER_PAGE : Int32 = 16

///|
const FUSED_QUALIFICATION_TOTAL_PAGES : Int32 = 4

///|
const FUSED_QUALIFICATION_CACHE_ELEMENTS : Int32 = 4096

///|
const FUSED_QUALIFICATION_POISON : Int32 = 0x42b6

///|
pub fn qualification_block_threads() -> Int32 {
  FUSED_QUALIFICATION_BLOCK_THREADS
}

///|
pub fn qualification_input_width() -> Int32 {
  FUSED_QUALIFICATION_INPUT_WIDTH
}

///|
pub fn qualification_query_heads() -> Int32 {
  FUSED_QUALIFICATION_QUERY_HEADS
}

///|
pub fn qualification_kv_heads() -> Int32 {
  FUSED_QUALIFICATION_KV_HEADS
}

///|
pub fn qualification_head_dimension() -> Int32 {
  FUSED_QUALIFICATION_HEAD_DIMENSION
}

///|
pub fn qualification_output_width() -> Int32 {
  FUSED_QUALIFICATION_OUTPUT_WIDTH
}

///|
pub fn qualification_tokens_per_page() -> Int32 {
  FUSED_QUALIFICATION_TOKENS_PER_PAGE
}

///|
pub fn qualification_total_pages() -> Int32 {
  FUSED_QUALIFICATION_TOTAL_PAGES
}

///|
pub fn qualification_cache_elements() -> Int32 {
  FUSED_QUALIFICATION_CACHE_ELEMENTS
}

///|
pub fn qualification_poison_bf16() -> Int32 {
  FUSED_QUALIFICATION_POISON
}
