#ifndef LUNAFLUX_CUDA_ORDERED_EXECUTOR_INTERNAL_H
#define LUNAFLUX_CUDA_ORDERED_EXECUTOR_INTERNAL_H

#include "resource_internal.h"

#include <stdatomic.h>
#include <stdint.h>

#define LF_MAX_KERNEL_ARGUMENTS 32
#define LF_DIMENSIONS_PER_KERNEL 7
#define LF_MAX_ORDERED_KERNELS 65536
#define LF_MAX_ORDERED_ARGUMENTS \
  (LF_MAX_ORDERED_KERNELS * LF_MAX_KERNEL_ARGUMENTS)

enum lf_ordered_phase {
  LF_ORDERED_IDLE = 0,
  LF_ORDERED_ENQUEUED = 1,
  LF_ORDERED_RECORDED = 2,
  LF_ORDERED_COMPLETE = 3
};

enum lf_ordered_policy {
  LF_ORDERED_EAGER_ONLY = 0,
  LF_ORDERED_CAPTURE_REQUIRED = 1,
  LF_ORDERED_CAPTURE_WITH_EAGER_FALLBACK = 2
};

enum lf_ordered_mode {
  LF_ORDERED_MODE_EAGER = 0,
  LF_ORDERED_MODE_CAPTURED = 1
};

typedef struct lf_ordered_kernel {
  lf_function *function;
  lf_allocation **allocations;
  CUdeviceptr *argument_values;
  void **kernel_parameters;
  int32_t argument_count;
  int32_t dimensions[LF_DIMENSIONS_PER_KERNEL];
} lf_ordered_kernel;

struct lf_ordered_executor {
  lf_context *context;
  lf_child *stream;
  CUevent event;
  CUgraph graph;
  CUgraphExec graph_exec;
  lf_ordered_kernel *kernels;
  int32_t kernel_count;
  int32_t acquired_kernel_count;
  int32_t execution_mode;
  atomic_int next_kernel;
  atomic_int phase;
  atomic_int state;
  atomic_int active_operations;
  atomic_int operation_gate;
};

int32_t lf_ordered_graph_prepare(
  lf_ordered_executor *executor,
  int32_t policy
);
int32_t lf_ordered_graph_launch(lf_ordered_executor *executor);
int32_t lf_ordered_graph_destroy(lf_ordered_executor *executor);
int32_t lf_ordered_operation_begin(lf_ordered_executor *executor);
void lf_ordered_operation_end(lf_ordered_executor *executor);

#endif
