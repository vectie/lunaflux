#include "resource_internal.h"

#include <stdint.h>
#include <string.h>

#define LF_FAKE_CONTEXT ((void *)(uintptr_t)0x100)
#define LF_FAKE_STREAM ((void *)(uintptr_t)0x200)
#define LF_FAKE_CUBLAS ((void *)(uintptr_t)0x300)
#define LF_FAKE_LEFT ((CUdeviceptr)0x1000)
#define LF_FAKE_RIGHT ((CUdeviceptr)0x2000)
#define LF_FAKE_OUTPUT ((CUdeviceptr)0x3000)
#define LF_FAKE_WORKSPACE ((CUdeviceptr)0x4000)

typedef struct lf_probe_state {
  int32_t layout_creates;
  int32_t layout_destroys;
  int32_t operation_creates;
  int32_t operation_destroys;
  int32_t fail_layout_create_at;
  int32_t fail_destroy_once;
  int32_t fail_mem_free_once;
  int32_t copy_to_device_calls;
  int32_t copy_to_host_calls;
  int32_t matmul_calls;
  int32_t stream_synchronize_calls;
  const uint8_t *expected_host_source;
} lf_probe_state;

static lf_probe_state *lf_active_probe;

static CUresult lf_fake_success_context(CUcontext context) {
  return context == LF_FAKE_CONTEXT ? 0 : 1;
}

static CUresult lf_fake_context_destroy(CUcontext context) {
  return context == LF_FAKE_CONTEXT ? 0 : 1;
}

static CUresult lf_fake_stream_destroy(CUstream stream) {
  return stream == LF_FAKE_STREAM ? 0 : 1;
}

static CUresult lf_fake_stream_synchronize(CUstream stream) {
  if (stream != LF_FAKE_STREAM) return 1;
  lf_active_probe->stream_synchronize_calls += 1;
  return 0;
}

static CUresult lf_fake_mem_free(CUdeviceptr address) {
  if (lf_active_probe->fail_mem_free_once != 0) {
    lf_active_probe->fail_mem_free_once = 0;
    return 1;
  }
  return address != 0 ? 0 : 1;
}

static CUresult lf_fake_copy_to_device(
  CUdeviceptr destination,
  const void *source,
  size_t byte_count
) {
  if (destination != LF_FAKE_LEFT + 3 ||
      source != lf_active_probe->expected_host_source + 5 ||
      byte_count != 8) return 1;
  lf_active_probe->copy_to_device_calls += 1;
  return 0;
}

static CUresult lf_fake_copy_to_host(
  void *destination,
  CUdeviceptr source,
  size_t byte_count
) {
  if (source != LF_FAKE_LEFT + 4 || byte_count != 6) return 1;
  uint8_t *bytes = (uint8_t *)destination;
  for (size_t index = 0; index < byte_count; index += 1) {
    bytes[index] = (uint8_t)(index + 10);
  }
  lf_active_probe->copy_to_host_calls += 1;
  return 0;
}

static int32_t lf_fake_cublas_destroy(cublasLtHandle_t handle) {
  return handle == LF_FAKE_CUBLAS ? 0 : 1;
}

static int32_t lf_fake_operation_create(
  cublasLtMatmulDesc_t *operation,
  int32_t compute_type,
  int32_t scale_type
) {
  if (compute_type != 68 || scale_type != 0) return 1;
  lf_active_probe->operation_creates += 1;
  *operation = (void *)(uintptr_t)0x500;
  return 0;
}

static int32_t lf_fake_operation_destroy(cublasLtMatmulDesc_t operation) {
  if (lf_active_probe->fail_destroy_once != 0) {
    lf_active_probe->fail_destroy_once = 0;
    return 1;
  }
  if (operation != (void *)(uintptr_t)0x500) return 1;
  lf_active_probe->operation_destroys += 1;
  return 0;
}

static int32_t lf_fake_layout_create(
  cublasLtMatrixLayout_t *layout,
  int32_t data_type,
  uint64_t rows,
  uint64_t columns,
  int64_t leading_dimension
) {
  lf_active_probe->layout_creates += 1;
  int32_t index = lf_active_probe->layout_creates;
  int expected_shape =
    (index == 1 && rows == 3 && columns == 4 && leading_dimension == 3) ||
    (index == 2 && rows == 4 && columns == 2 && leading_dimension == 4) ||
    (index == 3 && rows == 3 && columns == 2 && leading_dimension == 3);
  if (data_type != 14 || !expected_shape ||
      lf_active_probe->layout_creates ==
        lf_active_probe->fail_layout_create_at) return 1;
  *layout = (void *)(uintptr_t)(0x600 + lf_active_probe->layout_creates);
  return 0;
}

static int32_t lf_fake_layout_destroy(cublasLtMatrixLayout_t layout) {
  if (lf_active_probe->fail_destroy_once != 0) {
    lf_active_probe->fail_destroy_once = 0;
    return 1;
  }
  if ((uintptr_t)layout < 0x601 || (uintptr_t)layout > 0x610) return 1;
  lf_active_probe->layout_destroys += 1;
  return 0;
}

static int32_t lf_fake_matmul(
  cublasLtHandle_t handle,
  cublasLtMatmulDesc_t operation,
  const void *alpha,
  const void *first,
  cublasLtMatrixLayout_t first_layout,
  const void *second,
  cublasLtMatrixLayout_t second_layout,
  const void *beta,
  const void *input_output,
  cublasLtMatrixLayout_t input_output_layout,
  void *output,
  cublasLtMatrixLayout_t output_layout,
  const void *algorithm,
  void *workspace,
  size_t workspace_size,
  CUstream stream
) {
  if (handle != LF_FAKE_CUBLAS || operation == NULL ||
      *(const float *)alpha != 1.0f || *(const float *)beta != 0.0f ||
      first != (const void *)(uintptr_t)LF_FAKE_RIGHT ||
      first_layout != (void *)(uintptr_t)0x601 ||
      second != (const void *)(uintptr_t)LF_FAKE_LEFT ||
      second_layout != (void *)(uintptr_t)0x602 ||
      input_output != (const void *)(uintptr_t)LF_FAKE_OUTPUT ||
      output != (void *)(uintptr_t)LF_FAKE_OUTPUT ||
      input_output_layout != (void *)(uintptr_t)0x603 ||
      output_layout != input_output_layout ||
      algorithm != NULL ||
      workspace != (void *)(uintptr_t)LF_FAKE_WORKSPACE ||
      workspace_size != 512 || stream != LF_FAKE_STREAM) return 1;
  lf_active_probe->matmul_calls += 1;
  return 0;
}

static void lf_fake_finalizer(void *object) {
  (void)object;
}

static lf_context *lf_make_context(lf_cuda_api *api) {
  lf_context *context = (lf_context *)moonbit_make_external_object(
    lf_fake_finalizer,
    sizeof(lf_context)
  );
  memset(context, 0, sizeof(*context));
  context->api = api;
  context->handle = LF_FAKE_CONTEXT;
  atomic_init(&context->state, LF_RESOURCE_LIVE);
  atomic_init(&context->active_operations, 0);
  atomic_init(&context->children, 0);
  return context;
}

static lf_child *lf_make_child(
  lf_context *context,
  void *handle,
  int retain_context
) {
  lf_child *child = (lf_child *)moonbit_make_external_object(
    lf_fake_finalizer,
    sizeof(lf_child)
  );
  memset(child, 0, sizeof(*child));
  child->context = context;
  child->handle = handle;
  atomic_init(&child->state, LF_RESOURCE_LIVE);
  atomic_init(&child->active_operations, 0);
  atomic_init(&child->children, 0);
  if (retain_context != 0) {
    moonbit_incref(context);
    atomic_fetch_add(&context->children, 1);
  }
  return child;
}

static lf_allocation *lf_make_allocation(
  lf_context *context,
  CUdeviceptr handle,
  size_t size,
  int retain_context
) {
  lf_allocation *allocation = (lf_allocation *)moonbit_make_external_object(
    lf_fake_finalizer,
    sizeof(lf_allocation)
  );
  memset(allocation, 0, sizeof(*allocation));
  allocation->context = context;
  allocation->handle = handle;
  allocation->size = size;
  atomic_init(&allocation->state, LF_RESOURCE_LIVE);
  atomic_init(&allocation->active_operations, 0);
  if (retain_context != 0) {
    moonbit_incref(context);
    atomic_fetch_add(&context->children, 1);
  }
  return allocation;
}

static void lf_initialize_fake_api(lf_cuda_api *api) {
  memset(api, 0, sizeof(*api));
  api->cuCtxSetCurrent = lf_fake_success_context;
  api->cuCtxDestroy = lf_fake_context_destroy;
  api->cuStreamDestroy = lf_fake_stream_destroy;
  api->cuStreamSynchronize = lf_fake_stream_synchronize;
  api->cuMemFree = lf_fake_mem_free;
  api->cuMemcpyHtoD = lf_fake_copy_to_device;
  api->cuMemcpyDtoH = lf_fake_copy_to_host;
  api->cublasLtDestroy = lf_fake_cublas_destroy;
  api->cublasLtMatmulDescCreate = lf_fake_operation_create;
  api->cublasLtMatmulDescDestroy = lf_fake_operation_destroy;
  api->cublasLtMatrixLayoutCreate = lf_fake_layout_create;
  api->cublasLtMatrixLayoutDestroy = lf_fake_layout_destroy;
  api->cublasLtMatmul = lf_fake_matmul;
}

static int32_t lf_test_transfer_bounds(void) {
  lf_probe_state state;
  memset(&state, 0, sizeof(state));
  lf_active_probe = &state;
  lf_cuda_api api;
  lf_initialize_fake_api(&api);
  lf_context *context = lf_make_context(&api);
  lf_allocation *allocation = lf_make_allocation(
    context,
    LF_FAKE_LEFT,
    16,
    0
  );
  moonbit_bytes_t source = moonbit_make_bytes(32, 0);
  for (int32_t index = 0; index < 32; index += 1) source[index] = (uint8_t)index;
  state.expected_host_source = source;
  int32_t result = 0;
  if (lunaflux_cuda_copy_to_device(allocation, source, 5, 3, 8) != LF_OK ||
      state.copy_to_device_calls != 1) result = 101;
  if (result == 0 &&
      lunaflux_cuda_copy_to_device(allocation, source, 25, 0, 8) !=
        LF_INVALID_ARGUMENT) result = 102;
  if (result == 0 &&
      lunaflux_cuda_copy_to_device(allocation, source, 0, 9, 8) !=
        LF_INVALID_ARGUMENT) result = 103;
  if (result == 0 && state.copy_to_device_calls != 1) result = 104;
  int32_t status = 0;
  moonbit_bytes_t output = lunaflux_cuda_copy_to_host(allocation, 4, 6, &status);
  if (result == 0 &&
      (status != LF_OK || Moonbit_array_length(output) != 6 ||
       output[0] != 10 || output[5] != 15 ||
       state.copy_to_host_calls != 1)) result = 105;
  moonbit_decref(output);
  output = lunaflux_cuda_copy_to_host(allocation, 12, 6, &status);
  if (result == 0 &&
      (status != LF_INVALID_ARGUMENT || Moonbit_array_length(output) != 0 ||
       state.copy_to_host_calls != 1)) result = 106;
  moonbit_decref(output);
  moonbit_decref(source);
  moonbit_decref(allocation);
  moonbit_decref(context);
  lf_active_probe = NULL;
  return result;
}

static int32_t lf_test_allocation_retry(void) {
  lf_probe_state state;
  memset(&state, 0, sizeof(state));
  state.fail_mem_free_once = 1;
  lf_active_probe = &state;
  lf_cuda_api api;
  lf_initialize_fake_api(&api);
  lf_context *context = lf_make_context(&api);
  lf_allocation *allocation = lf_make_allocation(
    context,
    LF_FAKE_LEFT,
    16,
    1
  );
  int32_t result = 0;
  if (lunaflux_cuda_allocation_close(allocation) != LF_DRIVER_FAILURE ||
      atomic_load(&allocation->state) != LF_RESOURCE_LIVE ||
      allocation->handle != LF_FAKE_LEFT ||
      atomic_load(&context->children) != 1) result = 201;
  if (result == 0 && lunaflux_cuda_allocation_close(allocation) != LF_OK) {
    result = 202;
  }
  if (result == 0 &&
      (atomic_load(&allocation->state) != LF_RESOURCE_CLOSED ||
       allocation->handle != 0 || atomic_load(&context->children) != 0)) {
    result = 203;
  }
  if (result == 0 && lunaflux_cuda_context_close(context) != LF_OK) result = 204;
  moonbit_decref(allocation);
  moonbit_decref(context);
  lf_active_probe = NULL;
  return result;
}

static int32_t lf_test_gemm_lifecycle(void) {
  lf_probe_state state;
  memset(&state, 0, sizeof(state));
  lf_active_probe = &state;
  lf_cuda_api api;
  lf_initialize_fake_api(&api);
  lf_context *context = lf_make_context(&api);
  lf_child *cublas = lf_make_child(context, LF_FAKE_CUBLAS, 1);
  lf_child *stream = lf_make_child(context, LF_FAKE_STREAM, 1);
  lf_allocation *left = lf_make_allocation(context, LF_FAKE_LEFT, 16, 1);
  lf_allocation *right = lf_make_allocation(context, LF_FAKE_RIGHT, 24, 1);
  lf_allocation *output = lf_make_allocation(context, LF_FAKE_OUTPUT, 16, 1);
  lf_allocation *workspace = lf_make_allocation(
    context, LF_FAKE_WORKSPACE, 1024, 1
  );
  int32_t result = 0;
  int32_t status = 0;
  state.fail_layout_create_at = 2;
  lf_gemm_plan *failed = lunaflux_cuda_bf16_gemm_plan_create(
    cublas, 2, 4, 3, 512, &status
  );
  if (status != LF_DRIVER_FAILURE || atomic_load(&cublas->children) != 0 ||
      state.operation_creates != state.operation_destroys ||
      state.layout_creates - 1 != state.layout_destroys) result = 301;
  moonbit_decref(failed);
  state.layout_creates = 0;
  state.layout_destroys = 0;
  state.operation_creates = 0;
  state.operation_destroys = 0;
  state.fail_layout_create_at = 2;
  state.fail_destroy_once = 1;
  failed = lunaflux_cuda_bf16_gemm_plan_create(
    cublas, 2, 4, 3, 512, &status
  );
  if (result == 0 &&
      (status != LF_DRIVER_FAILURE ||
       atomic_load(&failed->state) != LF_RESOURCE_LIVE ||
       atomic_load(&cublas->children) != 1)) {
    result = 321;
  }
  moonbit_decref(failed);
  if (result == 0 &&
      (atomic_load(&cublas->children) != 0 || state.operation_creates !=
       state.operation_destroys ||
       state.layout_creates - 1 != state.layout_destroys)) {
    result = 322;
  }
  state.fail_layout_create_at = 0;
  state.layout_creates = 0;
  state.layout_destroys = 0;
  state.operation_creates = 0;
  state.operation_destroys = 0;
  lf_gemm_plan *plan = lunaflux_cuda_bf16_gemm_plan_create(
    cublas,
    2,
    4,
    3,
    512,
    &status
  );
  if (result == 0 &&
      (status != LF_OK || atomic_load(&cublas->children) != 1)) result = 302;
  if (result == 0 && lunaflux_cuda_cublas_close(cublas) != LF_BUSY) {
    result = 303;
  }
  if (result == 0 &&
      lunaflux_cuda_bf16_gemm_plan_run(
        plan,
        left,
        0,
        right,
        0,
        output,
        0,
        workspace,
        0,
        stream
      ) != LF_OK) result = 304;
  if (result == 0 &&
      (state.matmul_calls != 1 || state.stream_synchronize_calls != 1)) {
    result = 305;
  }
  if (result == 0 &&
      lunaflux_cuda_bf16_gemm_plan_run(
        plan,
        left,
        16,
        right,
        0,
        output,
        0,
        workspace,
        0,
        stream
      ) != LF_INVALID_ARGUMENT) result = 306;
  if (result == 0 &&
      lunaflux_cuda_bf16_gemm_plan_run(
        plan,
        left,
        0,
        right,
        0,
        output,
        0,
        workspace,
        1,
        stream
      ) != LF_INVALID_ARGUMENT) result = 307;
  if (result == 0 &&
      (state.matmul_calls != 1 || state.stream_synchronize_calls != 1)) {
    result = 308;
  }
  state.fail_destroy_once = 1;
  if (result == 0 &&
      lunaflux_cuda_bf16_gemm_plan_close(plan) != LF_DRIVER_FAILURE) {
    result = 309;
  }
  if (result == 0 &&
      (atomic_load(&plan->state) != LF_RESOURCE_LIVE ||
       atomic_load(&cublas->children) != 1)) result = 310;
  if (result == 0 && lunaflux_cuda_bf16_gemm_plan_close(plan) != LF_OK) {
    result = 311;
  }
  if (result == 0 &&
      (atomic_load(&plan->state) != LF_RESOURCE_CLOSED ||
       atomic_load(&cublas->children) != 0 ||
       state.layout_destroys != 3 || state.operation_destroys != 1)) {
    result = 312;
  }
  if (result == 0 && lunaflux_cuda_cublas_close(cublas) != LF_OK) result = 313;
  if (result == 0 && lunaflux_cuda_stream_close(stream) != LF_OK) result = 314;
  if (result == 0 && lunaflux_cuda_allocation_close(left) != LF_OK) result = 315;
  if (result == 0 && lunaflux_cuda_allocation_close(right) != LF_OK) result = 316;
  if (result == 0 && lunaflux_cuda_allocation_close(output) != LF_OK) result = 317;
  if (result == 0 && lunaflux_cuda_allocation_close(workspace) != LF_OK) {
    result = 318;
  }
  if (result == 0 && atomic_load(&context->children) != 0) result = 319;
  if (result == 0 && lunaflux_cuda_context_close(context) != LF_OK) result = 320;
  moonbit_decref(plan);
  moonbit_decref(workspace);
  moonbit_decref(output);
  moonbit_decref(right);
  moonbit_decref(left);
  moonbit_decref(stream);
  moonbit_decref(cublas);
  moonbit_decref(context);
  lf_active_probe = NULL;
  return result;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_test_execution_boundary(int32_t cycles) {
  if (cycles < 1 || cycles > 10000) return LF_INVALID_ARGUMENT;
  for (int32_t cycle = 0; cycle < cycles; cycle += 1) {
    int32_t result = lf_test_transfer_bounds();
    if (result != 0) return result;
    result = lf_test_allocation_retry();
    if (result != 0) return result;
    result = lf_test_gemm_lifecycle();
    if (result != 0) return result;
  }
  return LF_OK;
}
