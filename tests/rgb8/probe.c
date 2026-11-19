/* Exact production decode.c + a test byte ABI. No MoonBit runtime or GPU. */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static size_t codec_live;
static void *codec_malloc(size_t bytes) {
  void *p = malloc(bytes);
  if (p) codec_live++;
  return p;
}
static void codec_free(void *p) {
  if (p) { assert(codec_live > 0); codec_live--; free(p); }
}
static void *codec_realloc(void *p, size_t bytes) {
  if (bytes == 0) { codec_free(p); return NULL; }
  void *result = realloc(p, bytes);
  if (result && !p) codec_live++;
  return result;
}
#define STBI_MALLOC codec_malloc
#define STBI_REALLOC codec_realloc
#define STBI_FREE codec_free
#include "../../media/rgb8/decode.c"

static moonbit_bytes_t file_bytes(const char *directory, const char *name) {
  char path[4096];
  assert(snprintf(path, sizeof(path), "%s/%s", directory, name) > 0);
  FILE *file = fopen(path, "rb");
  assert(file);
  assert(fseek(file, 0, SEEK_END) == 0);
  long size = ftell(file);
  assert(size >= 0 && size <= 4096);
  rewind(file);
  moonbit_bytes_t bytes = moonbit_make_bytes((int32_t)size, 0);
  assert(fread(bytes, 1, (size_t)size, file) == (size_t)size);
  assert(fclose(file) == 0);
  return bytes;
}

int main(int argc, char **argv) {
  assert(argc == 2);
  const char *names[] = {"rgb.png", "rgba.png", "gray.png", "baseline.jpg", "progressive.jpg", "rgb16.png", "progressive-detail.jpg"};
  for (size_t image = 0; image < sizeof(names) / sizeof(names[0]); image++) {
    moonbit_bytes_t encoded = file_bytes(argv[1], names[image]);
    int32_t dimensions[3];
    const int maximum_pixels = image == 6 ? 187 : 64;
    moonbit_bytes_t rgb = lunaflux_rgb8_decode(encoded, maximum_pixels, dimensions);
    assert(codec_live == 0);
    if (image == 5) {
      assert(dimensions[2] == 3 && Moonbit_array_length(rgb) == 0);
    } else {
      assert(dimensions[2] == 0);
      int width = image == 6 ? 17 : image >= 3 ? 8 : 2;
      int height = image == 6 ? 11 : image >= 3 ? 8 : 1;
      assert(dimensions[0] == width && dimensions[1] == height);
      assert(Moonbit_array_length(rgb) == width * height * 3);
      if (image < 2) {
        const unsigned char expected[] = {255, 0, 0, 0, 255, 0};
        assert(memcmp(rgb, expected, sizeof(expected)) == 0);
      } else if (image == 2) {
        const unsigned char expected[] = {0, 0, 0, 128, 128, 128};
        assert(memcmp(rgb, expected, sizeof(expected)) == 0);
      } else if (image == 6) {
        moonbit_bytes_t expected = file_bytes(argv[1], "progressive-detail.rgb");
        assert(Moonbit_array_length(expected) == 561);
        for (int i = 0; i < 561; ++i) assert(abs((int)rgb[i] - expected[i]) <= 2);
        release_bytes(expected);
      } else {
        for (int i = 0; i < 64; i++) {
          assert(rgb[i * 3] >= 253 && rgb[i * 3 + 1] <= 1 && rgb[i * 3 + 2] <= 1);
        }
      }
    }
    release_bytes(rgb);
    rgb = lunaflux_rgb8_decode(encoded, 1, dimensions);
    assert(dimensions[2] == 2 && Moonbit_array_length(rgb) == 0 && codec_live == 0);
    release_bytes(rgb);
    /* Every truncated prefix must return safely and release codec temporaries. */
    for (int cut = 0; cut < Moonbit_array_length(encoded); cut++) {
      moonbit_bytes_t prefix = moonbit_make_bytes(cut, 0);
      memcpy(prefix, encoded, (size_t)cut);
      rgb = lunaflux_rgb8_decode(prefix, maximum_pixels, dimensions);
      assert(codec_live == 0);
      release_bytes(rgb);
      release_bytes(prefix);
    }
    release_bytes(encoded);
  }
  puts("RGB8 PNG/JPEG: colors/alpha/grayscale, baseline/progressive and buffered-entropy regression, precision/size rejection and all truncated prefixes passed; codec allocation count balanced");
  return 0;
}
