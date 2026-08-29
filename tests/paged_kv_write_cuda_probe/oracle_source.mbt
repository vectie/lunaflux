///|
const KV_SERIAL_ORACLE_SYMBOL : String = "lunaflux_paged_kv_write_serial_oracle_v1"

///|
fn kv_serial_oracle_source() -> Bytes {
  @utf8.encode(
    "#include <cuda_bf16.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void lunaflux_paged_kv_write_serial_oracle_v1(\n" +
    " const int32_t* step_counts, const int32_t* query_positions,\n" +
    " const int32_t* query_row_offsets, const int32_t* sequence_lengths,\n" +
    " const int32_t* page_table_offsets, const int32_t* page_table_page_indices,\n" +
    " const __nv_bfloat16* positioned_qkv, __nv_bfloat16* key_cache,\n" +
    " __nv_bfloat16* value_cache) {\n" +
    " if (blockIdx.x || blockIdx.y || blockIdx.z || threadIdx.x || blockDim.x != 1) return;\n" +
    " const int prefill=step_counts[0], decode=step_counts[1], rows=step_counts[2];\n" +
    " const int tokens=step_counts[3], cells=step_counts[4];\n" +
    " if(prefill<0||decode<0||(long long)rows!=(long long)prefill+(long long)decode||rows<=0||rows>4||tokens<=0||tokens>8||cells<=0||cells>16)return;\n" +
    " if(query_row_offsets[0]!=0||query_row_offsets[rows]!=tokens||page_table_offsets[0]!=0||page_table_offsets[rows]!=cells)return;\n" +
    " for(int r=0;r<rows;++r){if(query_row_offsets[r]<0||query_row_offsets[r]>=query_row_offsets[r+1]||query_row_offsets[r+1]>tokens||sequence_lengths[r]<=0||page_table_offsets[r]<0||page_table_offsets[r]>=page_table_offsets[r+1]||page_table_offsets[r+1]>cells)return;}\n" +
    " for(int token=0;token<tokens;++token){int row=0;while(row+1<rows&&token>=query_row_offsets[row+1])++row;\n" +
    "  const int position=query_positions[token];if(position<0||position>=sequence_lengths[row])return;\n" +
    "  const int begin=page_table_offsets[row],end=page_table_offsets[row+1];\n" +
    "  const int logical_page=position/16;if(logical_page<0||logical_page>=end-begin)return;\n" +
    "  const int cell=begin+logical_page;if(cell>=cells)return;\n" +
    "  const int page=page_table_page_indices[cell];if(page<0||page>=64)return;\n" +
    "  for(int component=0;component<64;++component){\n" +
    "   const unsigned long long cache=(unsigned long long)page*1024ULL+(unsigned long long)(position%16)*64ULL+component;\n" +
    "   const unsigned long long qkv=(unsigned long long)token*256ULL;\n" +
    "   key_cache[cache]=positioned_qkv[qkv+128ULL+component];\n" +
    "   value_cache[cache]=positioned_qkv[qkv+192ULL+component];\n" +
    "  }\n }\n}\n",
  )
}
