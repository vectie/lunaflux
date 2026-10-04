///|
/// The wrapper translates CSR page metadata by value. It neither gathers KV
/// nor allocates a per-step host plan. The reference's device function owns
/// its asynchronous pipeline and reductions unchanged.
fn flashinfer_source(plan : ReferenceAttentionPlan) -> String {
  let (heads, kv_heads, dimension, page) = plan.abi.shape()
  let (width, stride, _) = plan.abi.layout()
  let (rows, _, _) = plan.abi.limits()
  let group = heads / kv_heads
  let z = 128 / (16 * group)
  let source =
    $|#include <flashinfer/attention/decode.cuh>
    $|#include <flashinfer/attention/variants.cuh>
    $|#include <flashinfer/fastdiv.cuh>
    $|struct LfPageSize {
    $| __host__ __device__ operator uint32_t() const { return \{page}; }
    $| __host__ __device__ void divmod(uint32_t n, uint32_t &q, uint32_t &r) const { q=n/\{page}; r=n%\{page}; }
    $|};
    $|struct LfReferencePages {
    $| const __nv_bfloat16 *k_data, *v_data;
    $| const int *indptr, *indices, *lengths;
    $| const int *rope_pos_offset=nullptr;
    $| uint32_t batch_size=1, num_heads=\{kv_heads};
    $| LfPageSize page_size;
    $| __host__ __device__ uint32_t get_length(uint32_t) const { return lengths[0]; }
    $| __device__ size_t protective_get_kv_offset(uint32_t entry, uint32_t head, uint32_t token, uint32_t component, int last) const {
    $|   return entry < (uint32_t)last ? (size_t)indices[entry]*\{stride}ULL+(token*\{kv_heads}+head)*\{dimension}+component : 0;
    $| }
    $|};
    $|struct LfReferenceParams {
    $| using DTypeQ=__nv_bfloat16; using DTypeKV=__nv_bfloat16; using DTypeO=__nv_bfloat16; using IdType=int;
    $| const DTypeQ *q; DTypeO *o; float *lse=nullptr;
    $| LfReferencePages paged_kv;
    $| const int *request_indices, *kv_tile_indices, *kv_chunk_size_ptr;
    $| const bool *block_valid_mask=nullptr;
    $| uint32_t padded_batch_size=1, num_qo_heads=\{heads}, q_stride_n=\{width}, q_stride_h=\{dimension};
    $| bool partition_kv=false;
    $| int window_left=-1;
    $| float sm_scale=0.08838834764831845f;
    $| __host__ __device__ uint32_t get_qo_len(uint32_t) const { return 1; }
    $| __host__ __device__ uint32_t get_kv_len(uint32_t row) const { return paged_kv.get_length(row); }
    $|};
    $|extern "C" __global__ void \{plan.symbol}(
    $| const int *counts, const int *positions, const int *offsets,
    $| const int *lengths, const int *page_offsets, const int *page_indices,
    $| const __nv_bfloat16 *input, __nv_bfloat16 *output,
    $| const __nv_bfloat16 *keys, const __nv_bfloat16 *values) {
    $| const int row=counts[0]+blockIdx.x;
    $| if (blockDim.x != 16 || blockDim.y != \{group} || blockDim.z != \{z} || row>=counts[2] || counts[2]>\{rows}) return;
    $| const int token=offsets[row];
    $| if (offsets[row+1]-token != 1 || positions[token] != lengths[row]-1) return;
    $| int zero=0, chunk=lengths[row];
    $| LfReferenceParams p{};
    $| p.q=input+(unsigned long long)token*\{width}; p.o=output+(unsigned long long)token*\{heads * dimension};
    $| p.paged_kv.k_data=keys; p.paged_kv.v_data=values;
    $| p.paged_kv.indptr=page_offsets+row; p.paged_kv.indices=page_indices; p.paged_kv.lengths=lengths+row;
    $| p.request_indices=&zero; p.kv_tile_indices=&zero; p.kv_chunk_size_ptr=&chunk;
    $| extern __shared__ uint8_t smem[];
    $| using Variant=flashinfer::DefaultAttention<false,false,false,false>;
    $| flashinfer::BatchDecodeWithPagedKVCacheDevice<flashinfer::PosEncodingMode::kNone,2,1,8,16,\{group},\{z},Variant>(p,smem,0,blockIdx.y);
    $|}
  source
}
