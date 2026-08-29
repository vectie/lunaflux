#include "nccl_abi.h"

#include <stdint.h>
#include <string.h>

#define LF_FAKE_VERSION 21403

typedef struct {
  int32_t init_calls;
  int32_t destroy_calls;
  int32_t abort_calls;
  int32_t live_handles;
  int32_t fail_init_once;
  int32_t fail_destroy_once;
  int32_t fail_abort_once;
} lf_nccl_probe_state;

static lf_nccl_probe_state *active_probe = NULL;

static lf_nccl_result lf_fake_get_version(int32_t *version) {
  *version = LF_FAKE_VERSION;
  return 0;
}

static lf_nccl_result lf_fake_get_unique_id(lf_nccl_unique_id *id) {
  for (int32_t index = 0; index < LF_NCCL_UNIQUE_ID_BYTES; index += 1) {
    id->internal[index] = (char)(index ^ 0x5a);
  }
  return 0;
}

static lf_nccl_result lf_fake_init_rank(
  lf_nccl_handle *handle,
  int32_t world_size,
  lf_nccl_unique_id id,
  int32_t rank,
  lf_nccl_config_v21400 *config
) {
  (void)id;
  if (active_probe == NULL || world_size <= 0 || rank < 0 ||
      rank >= world_size || config == NULL ||
      config->size != sizeof(lf_nccl_config_v21400) ||
      config->magic != LF_NCCL_CONFIG_MAGIC ||
      config->version != LF_NCCL_CONFIG_VERSION || config->blocking != 0) {
    return 1;
  }
  active_probe->init_calls += 1;
  *handle = (void *)(uintptr_t)(16 + active_probe->init_calls);
  active_probe->live_handles += 1;
  if (active_probe->fail_init_once != 0) {
    active_probe->fail_init_once = 0;
    return 1;
  }
  return 0;
}

static lf_nccl_result lf_fake_async_error(
  lf_nccl_handle handle,
  lf_nccl_result *status
) {
  if (handle == NULL || status == NULL) return 1;
  *status = 0;
  return 0;
}

static lf_nccl_result lf_fake_destroy(lf_nccl_handle handle) {
  if (active_probe == NULL || handle == NULL) return 1;
  active_probe->destroy_calls += 1;
  active_probe->live_handles -= 1;
  if (active_probe->fail_destroy_once != 0) {
    active_probe->fail_destroy_once = 0;
    return 1;
  }
  return 0;
}

static lf_nccl_result lf_fake_abort(lf_nccl_handle handle) {
  if (active_probe == NULL || handle == NULL) return 1;
  active_probe->abort_calls += 1;
  active_probe->live_handles -= 1;
  if (active_probe->fail_abort_once != 0) {
    active_probe->fail_abort_once = 0;
    return 1;
  }
  return 0;
}

static void lf_initialize_fake_api(lf_nccl_api *api) {
  memset(api, 0, sizeof(*api));
  api->availability = LF_NCCL_AVAILABLE;
  api->version_code = LF_FAKE_VERSION;
  api->get_version = lf_fake_get_version;
  api->get_unique_id = lf_fake_get_unique_id;
  api->comm_init_rank_config = lf_fake_init_rank;
  api->comm_get_async_error = lf_fake_async_error;
  api->comm_destroy = lf_fake_destroy;
  api->comm_abort = lf_fake_abort;
}

static lf_nccl_communicator *lf_fake_create(
  lf_nccl_api *api,
  const uint8_t *id,
  uint64_t generation,
  int32_t *status
) {
  lf_nccl_communicator *communicator = lf_nccl_communicator_create_with_api(
    api,
    LF_FAKE_VERSION,
    id,
    2,
    1,
    generation,
    0U,
    status
  );
  if (*status == LF_NCCL_OK) {
    int32_t ready = 0;
    *status = lf_nccl_communicator_poll_ready(communicator, &ready);
    if (*status == LF_NCCL_OK && ready != 1) {
      *status = LF_NCCL_RUNTIME_FAILURE;
    }
  }
  return communicator;
}

static int32_t lf_probe_scalar_commit(
  lf_nccl_communicator *communicator,
  uint64_t generation,
  uint64_t plan_sequence,
  uint64_t collective_sequence,
  int32_t operation_id,
  int32_t collective_kind
) {
  int32_t status = lf_nccl_collective_begin(
    communicator,
    generation,
    plan_sequence,
    collective_sequence,
    operation_id,
    collective_kind
  );
  if (status != LF_NCCL_OK) return status;
  lf_nccl_collective_commit(
    communicator,
    plan_sequence,
    collective_sequence,
    operation_id
  );
  return LF_NCCL_OK;
}

static int32_t lf_probe_cycle(void) {
  lf_nccl_api api;
  lf_initialize_fake_api(&api);
  lf_nccl_probe_state state;
  memset(&state, 0, sizeof(state));
  active_probe = &state;
  int32_t result = 0;
  int32_t version = 0;
  uint8_t id[LF_NCCL_UNIQUE_ID_BYTES];
  memset(id, 0x5a, sizeof(id));

  if (lf_nccl_runtime_admit_with_api(&api, 21403, 22999, &version) !=
        LF_NCCL_OK ||
      version != LF_FAKE_VERSION ||
      lf_nccl_runtime_admit_with_api(&api, 21404, 22999, &version) !=
        LF_NCCL_VERSION_MISMATCH) {
    result = 10;
  }
  if (result == 0 &&
      lf_nccl_unique_id_create_with_api(&api, LF_FAKE_VERSION, id) !=
        LF_NCCL_OK) {
    result = 11;
  }
  if (result == 0) {
    for (int32_t index = 0; index < LF_NCCL_UNIQUE_ID_BYTES; index += 1) {
      if (id[index] != (uint8_t)(index ^ 0x5a)) {
        result = 12;
        break;
      }
    }
  }

  int32_t status = 0;
  lf_nccl_communicator *small_world =
    lf_nccl_communicator_create_with_api(
      &api,
      LF_FAKE_VERSION,
      id,
      1,
      0,
      1U,
      0U,
      &status
    );
  if (result == 0 && status != LF_NCCL_INVALID_ARGUMENT) result = 13;
  moonbit_decref(small_world);
  lf_nccl_communicator *large_world =
    lf_nccl_communicator_create_with_api(
      &api,
      LF_FAKE_VERSION,
      id,
      17,
      0,
      1U,
      0U,
      &status
    );
  if (result == 0 && status != LF_NCCL_INVALID_ARGUMENT) result = 14;
  moonbit_decref(large_world);

  lf_nccl_communicator *healthy = lf_fake_create(&api, id, 7U, &status);
  if (result == 0 &&
      (status != LF_NCCL_OK || healthy->generation != 7U ||
       healthy->world_size != 2 || healthy->rank != 1)) {
    result = 20;
  }
  if (result == 0 &&
      lf_probe_scalar_commit(healthy, 7U, 1U, 1U, 0, 1) !=
        LF_NCCL_OK) {
    result = 21;
  }
  if (result == 0 &&
      lf_probe_scalar_commit(healthy, 7U, 1U, 2U, 5, 2) !=
        LF_NCCL_OK) {
    result = 22;
  }
  if (result == 0 &&
      lf_probe_scalar_commit(healthy, 7U, 2U, 3U, 0, 1) !=
        LF_NCCL_OK) {
    result = 23;
  }
  if (result == 0 && healthy->next_collective_sequence != 4U) result = 24;
  if (result == 0 && lf_nccl_communicator_close(healthy) != LF_NCCL_OK) {
    result = 25;
  }
  if (result == 0 && lf_nccl_communicator_close(healthy) != LF_NCCL_OK) {
    result = 26;
  }
  moonbit_decref(healthy);

  lf_nccl_communicator *invalidated = lf_fake_create(&api, id, 8U, &status);
  if (result == 0 && status != LF_NCCL_OK) result = 30;
  if (result == 0 &&
      lf_nccl_communicator_invalidate(invalidated, 8U) != LF_NCCL_OK) {
    result = 31;
  }
  if (result == 0 &&
      lf_nccl_communicator_close(invalidated) != LF_NCCL_FAILED) {
    result = 32;
  }
  if (result == 0 &&
      (lf_nccl_communicator_abort(invalidated) != LF_NCCL_OK ||
       lf_nccl_communicator_abort(invalidated) != LF_NCCL_OK)) {
    result = 33;
  }
  moonbit_decref(invalidated);

  lf_nccl_communicator *generation = lf_fake_create(&api, id, 9U, &status);
  if (result == 0 && status != LF_NCCL_OK) result = 40;
  if (result == 0 &&
      lf_probe_scalar_commit(generation, 10U, 1U, 1U, 0, 1) !=
        LF_NCCL_GENERATION_MISMATCH) {
    result = 41;
  }
  if (result == 0 && lf_nccl_communicator_abort(generation) != LF_NCCL_OK) {
    result = 42;
  }
  moonbit_decref(generation);

  lf_nccl_communicator *sequence = lf_fake_create(&api, id, 10U, &status);
  if (result == 0 && status != LF_NCCL_OK) result = 50;
  if (result == 0 &&
      lf_probe_scalar_commit(sequence, 10U, 1U, 0U, 0, 3) !=
        LF_NCCL_SEQUENCE_MISMATCH) {
    result = 51;
  }
  if (result == 0 && lf_nccl_communicator_abort(sequence) != LF_NCCL_OK) {
    result = 52;
  }
  moonbit_decref(sequence);

  lf_nccl_communicator *operation = lf_fake_create(&api, id, 11U, &status);
  if (result == 0 && status != LF_NCCL_OK) result = 55;
  if (result == 0 &&
      lf_probe_scalar_commit(operation, 11U, 1U, 1U, 5, 1) !=
        LF_NCCL_OK) {
    result = 56;
  }
  if (result == 0 &&
      lf_probe_scalar_commit(operation, 11U, 1U, 2U, 5, 1) !=
        LF_NCCL_SEQUENCE_MISMATCH) {
    result = 57;
  }
  if (result == 0 && lf_nccl_communicator_abort(operation) != LF_NCCL_OK) {
    result = 58;
  }
  moonbit_decref(operation);

  lf_nccl_communicator *operation_substitution =
    lf_fake_create(&api, id, 12U, &status);
  if (result == 0 && status != LF_NCCL_OK) result = 59;
  if (result == 0 &&
      lf_probe_scalar_commit(
        operation_substitution,
        12U,
        1U,
        1U,
        5,
        1
      ) != LF_NCCL_OK) {
    result = 60;
  }
  if (result == 0 &&
      lf_probe_scalar_commit(
        operation_substitution,
        12U,
        1U,
        2U,
        4,
        1
      ) != LF_NCCL_SEQUENCE_MISMATCH) {
    result = 61;
  }
  if (result == 0 &&
      lf_nccl_communicator_abort(operation_substitution) != LF_NCCL_OK) {
    result = 62;
  }
  moonbit_decref(operation_substitution);

  state.fail_destroy_once = 1;
  lf_nccl_communicator *destroy_failure = lf_fake_create(&api, id, 13U, &status);
  if (result == 0 && status != LF_NCCL_OK) result = 60;
  if (result == 0 &&
      (lf_nccl_communicator_close(destroy_failure) !=
         LF_NCCL_RUNTIME_FAILURE ||
       lf_nccl_communicator_close(destroy_failure) != LF_NCCL_OK)) {
    result = 61;
  }
  moonbit_decref(destroy_failure);

  state.fail_abort_once = 1;
  lf_nccl_communicator *abort_failure = lf_fake_create(&api, id, 14U, &status);
  if (result == 0 && status != LF_NCCL_OK) result = 70;
  if (result == 0 &&
      (lf_nccl_communicator_abort(abort_failure) !=
         LF_NCCL_RUNTIME_FAILURE ||
       lf_nccl_communicator_abort(abort_failure) != LF_NCCL_OK)) {
    result = 71;
  }
  moonbit_decref(abort_failure);

  state.fail_init_once = 1;
  lf_nccl_communicator *init_failure = lf_fake_create(&api, id, 15U, &status);
  if (result == 0 && status != LF_NCCL_RUNTIME_FAILURE) result = 80;
  moonbit_decref(init_failure);

  if (result == 0 &&
      (state.init_calls != 9 || state.destroy_calls != 2 ||
       state.abort_calls != 7 || state.live_handles != 0)) {
    result = 90;
  }
  active_probe = NULL;
  return result;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_nccl_test_lifecycle(int32_t cycles) {
  if (cycles <= 0 || cycles > 1000000) return LF_NCCL_INVALID_ARGUMENT;
  for (int32_t cycle = 0; cycle < cycles; cycle += 1) {
    int32_t result = lf_probe_cycle();
    if (result != 0) return result;
  }
  return 0;
}
