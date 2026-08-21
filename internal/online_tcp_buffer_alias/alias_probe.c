#include <moonbit.h>

#include <stdint.h>
#include <stdlib.h>

moonbit_bytes_t lunaflux_online_tcp_retain_bytes_as_fixed_array(
  moonbit_bytes_t source
);

static int incref_calls = 0;
static int decref_calls = 0;

void moonbit_incref(void *object) {
  struct moonbit_object *header = Moonbit_object_header(object);
  int32_t count = Moonbit_rc_count(header);
  if (count <= 0 || count >= (int32_t)MOONBIT_RC_DYNAMIC_COUNT_MAX) abort();
  header->rc += (int32_t)(1u << MOONBIT_RC_COUNT_SHIFT);
  incref_calls += 1;
}

void moonbit_decref(void *object) {
  struct moonbit_object *header = Moonbit_object_header(object);
  int32_t count = Moonbit_rc_count(header);
  if (count <= 0) abort();
  header->rc -= (int32_t)(1u << MOONBIT_RC_COUNT_SHIFT);
  decref_calls += 1;
}

static moonbit_bytes_t make_bytes(int32_t length) {
  struct moonbit_object *header = calloc(
    1, sizeof(struct moonbit_object) + (size_t)length
  );
  if (header == NULL) return NULL;
  header->rc = Moonbit_make_dynamic_rc(moonbit_BLOCK_KIND_VAL_ARRAY);
  header->meta = (uint32_t)length;
  return (moonbit_bytes_t)(header + 1);
}

static int exercise_alias(int32_t length, uint8_t marker) {
  moonbit_bytes_t source = make_bytes(length);
  if (source == NULL) return 1;
  if (Moonbit_array_length(source) != length) return 2;
  if (Moonbit_rc_count(Moonbit_object_header(source)) != 1) return 3;

  int before_incref = incref_calls;
  moonbit_bytes_t alias =
    lunaflux_online_tcp_retain_bytes_as_fixed_array(source);
  if (alias != source) return 4;
  if (Moonbit_array_length(alias) != length) return 5;
  if (incref_calls != before_incref + 1) return 6;
  if (Moonbit_rc_count(Moonbit_object_header(source)) != 2) return 7;

  alias[0] = marker;
  if (source[0] != marker) return 8;
  alias[length - 1] = (uint8_t)(marker ^ UINT8_C(0xff));
  if (source[length - 1] != (uint8_t)(marker ^ UINT8_C(0xff))) return 9;

  moonbit_decref(alias);
  if (Moonbit_rc_count(Moonbit_object_header(source)) != 1) return 10;
  moonbit_decref(source);
  if (Moonbit_rc_count(Moonbit_object_header(source)) != 0) return 11;
  free(Moonbit_object_header(source));
  return 0;
}

int main(void) {
  const int32_t lengths[] = {1, 17, 4096};
  for (size_t index = 0; index < sizeof(lengths) / sizeof(lengths[0]);
       index += 1) {
    int result = exercise_alias(lengths[index], (uint8_t)(0x31u + index));
    if (result != 0) return result;
  }
  for (int32_t iteration = 0; iteration < 4096; iteration += 1) {
    int32_t length = 1 + iteration % 257;
    int result = exercise_alias(length, (uint8_t)iteration);
    if (result != 0) return 20 + result;
  }
  if (incref_calls != decref_calls / 2) return 40;
  return 0;
}
