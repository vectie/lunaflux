///|
const FUSED_QUALIFICATION_BLOCK_THREADS : Int = 128

///|
const FUSED_QUALIFICATION_INPUT_WIDTH : Int = 128

///|
const FUSED_QUALIFICATION_QUERY_HEADS : Int = 4

///|
const FUSED_QUALIFICATION_KV_HEADS : Int = 2

///|
const FUSED_QUALIFICATION_HEAD_DIMENSION : Int = 32

///|
const FUSED_QUALIFICATION_QUERY_WIDTH : Int = 128

///|
const FUSED_QUALIFICATION_KV_WIDTH : Int = 64

///|
const FUSED_QUALIFICATION_OUTPUT_WIDTH : Int = 256

///|
const FUSED_QUALIFICATION_TOKENS_PER_PAGE : Int = 16

///|
const FUSED_QUALIFICATION_TOTAL_PAGES : Int = 4

///|
const FUSED_QUALIFICATION_CACHE_ELEMENTS : Int = 4096

///|
const FUSED_QUALIFICATION_POISON : Int = 0x42b6

///|
pub fn qualification_block_threads() -> Int {
  FUSED_QUALIFICATION_BLOCK_THREADS
}

///|
pub fn qualification_input_width() -> Int {
  FUSED_QUALIFICATION_INPUT_WIDTH
}

///|
pub fn qualification_query_heads() -> Int {
  FUSED_QUALIFICATION_QUERY_HEADS
}

///|
pub fn qualification_kv_heads() -> Int {
  FUSED_QUALIFICATION_KV_HEADS
}

///|
pub fn qualification_head_dimension() -> Int {
  FUSED_QUALIFICATION_HEAD_DIMENSION
}

///|
pub fn qualification_output_width() -> Int {
  FUSED_QUALIFICATION_OUTPUT_WIDTH
}

///|
pub fn qualification_tokens_per_page() -> Int {
  FUSED_QUALIFICATION_TOKENS_PER_PAGE
}

///|
pub fn qualification_total_pages() -> Int {
  FUSED_QUALIFICATION_TOTAL_PAGES
}

///|
pub fn qualification_cache_elements() -> Int {
  FUSED_QUALIFICATION_CACHE_ELEMENTS
}

///|
pub fn qualification_poison_bf16() -> Int {
  FUSED_QUALIFICATION_POISON
}
