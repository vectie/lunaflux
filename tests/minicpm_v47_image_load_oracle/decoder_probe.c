// Offline isolated decoder diagnostic. Compile once with the normal SIMD path
// and once with STBI_NO_SIMD; neither executable is a runtime fallback.
#define _POSIX_C_SOURCE 200809L
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "stb_image.h"
#include <stdio.h>
#include <time.h>

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) return 2;
  int iterations = argc == 4 ? atoi(argv[3]) : 1;
  if (iterations <= 0) return 2;
  FILE* input = fopen(argv[1], "rb");
  if (!input) return 2;
  if (fseek(input, 0, SEEK_END)) { fclose(input); return 2; }
  long length = ftell(input);
  if (length <= 0 || length > 1048576 || fseek(input, 0, SEEK_SET)) { fclose(input); return 2; }
  unsigned char* encoded = malloc((size_t)length);
  if (!encoded) { fclose(input); return 2; }
  size_t read = fread(encoded, 1, (size_t)length, input);
  fclose(input);
  if (read != (size_t)length) { free(encoded); return 2; }
  int width = 0, height = 0, components = 0;
  unsigned char* pixels = NULL;
  struct timespec start, end;
  if (clock_gettime(CLOCK_MONOTONIC, &start)) { free(encoded); return 2; }
  for (int i = 0; i < iterations; ++i) {
    pixels = stbi_load_from_memory(encoded, (int)length, &width, &height, &components, 3);
    if (!pixels) { free(encoded); return 3; }
    if (i + 1 < iterations) stbi_image_free(pixels);
  }
  if (clock_gettime(CLOCK_MONOTONIC, &end)) { free(encoded); stbi_image_free(pixels); return 2; }
  free(encoded);
  FILE* output = fopen(argv[2], "wb");
  if (!output) { stbi_image_free(pixels); return 4; }
  size_t bytes = (size_t)width * height * 3;
  int result = fwrite(pixels, 1, bytes, output) == bytes ? 0 : 5;
  if (fclose(output)) result = 6;
  stbi_image_free(pixels);
  if (iterations > 1) {
    double elapsed = (end.tv_sec - start.tv_sec) * 1.0e9 + end.tv_nsec - start.tv_nsec;
    printf("{\"iterations\":%d,\"width\":%d,\"height\":%d,\"elapsed_ns\":%.0f,\"per_decode_ns\":%.3f}\n",
      iterations, width, height, elapsed, elapsed / iterations);
  }
  return result;
}
