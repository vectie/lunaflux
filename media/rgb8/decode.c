/* Narrow memory-only PNG/JPEG boundary. Temporary stb storage never escapes. */
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STBI_MAX_DIMENSIONS 32768
#include "stb_image.h"
#include <moonbit.h>
#include <stdint.h>
#include <string.h>

MOONBIT_FFI_EXPORT moonbit_bytes_t lunaflux_rgb8_decode(
    moonbit_bytes_t encoded, int32_t maximum_pixels, int32_t *dimensions) {
  int width = 0, height = 0, components = 0;
  int length = Moonbit_array_length(encoded);
  dimensions[0] = 0;
  dimensions[1] = 0;
  dimensions[2] = 1;
  if (length <= 0 || !stbi_info_from_memory(encoded, length, &width, &height,
                                           &components)) {
    return moonbit_make_bytes(0, 0);
  }
  int64_t pixels = (int64_t)width * (int64_t)height;
  if (width <= 0 || height <= 0 || pixels > maximum_pixels ||
      pixels > INT32_MAX / 3) {
    dimensions[2] = 2;
    return moonbit_make_bytes(0, 0);
  }
  if (stbi_is_16_bit_from_memory(encoded, length)) {
    dimensions[2] = 3;
    return moonbit_make_bytes(0, 0);
  }
  unsigned char *rgb = stbi_load_from_memory(encoded, length, &width, &height,
                                            &components, 3);
  if (rgb == NULL) return moonbit_make_bytes(0, 0);
  int32_t size = (int32_t)(pixels * 3);
  moonbit_bytes_t result = moonbit_make_bytes(size, 0);
  memcpy(result, rgb, (size_t)size);
  stbi_image_free(rgb);
  dimensions[0] = width;
  dimensions[1] = height;
  dimensions[2] = 0;
  return result;
}
