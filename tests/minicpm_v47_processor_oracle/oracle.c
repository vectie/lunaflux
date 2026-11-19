/* Offline RGB8 adapter for unchanged Pillow 12.0.0 Resample.c.
 * No first-party resize arithmetic lives in this oracle. */
#include "Imaging.h"

static void die(const char *message) {
    fprintf(stderr, "pillow RGB oracle: %s\n", message);
    exit(2);
}

Imaging ImagingNewDirty(ModeID mode, int width, int height) {
    if (mode != IMAGING_MODE_RGB || width <= 0 || height <= 0) die("geometry");
    Imaging image = calloc(1, sizeof(*image));
    if (!image) die("image allocation");
    image->mode = mode;
    image->type = IMAGING_TYPE_UINT8;
    image->bands = 3;
    image->xsize = width;
    image->ysize = height;
    image->pixelsize = 4;
    image->linesize = width * 4;
    image->block = calloc((size_t)height, (size_t)image->linesize);
    image->image = calloc((size_t)height, sizeof(char *));
    if (!image->block || !image->image) die("pixel allocation");
    for (int y = 0; y < height; ++y) image->image[y] = image->block + (size_t)y * image->linesize;
    image->image32 = (INT32 **)image->image;
    return image;
}

void ImagingDelete(Imaging image) {
    if (!image) return;
    free(image->image);
    free(image->block);
    free(image);
}

Imaging ImagingCopy(Imaging source) {
    Imaging image = ImagingNewDirty(source->mode, source->xsize, source->ysize);
    memcpy(image->block, source->block, (size_t)source->ysize * source->linesize);
    return image;
}

int isModeI16(const ModeID mode) {
    return mode == IMAGING_MODE_I_16 || mode == IMAGING_MODE_I_16L ||
           mode == IMAGING_MODE_I_16B || mode == IMAGING_MODE_I_16N;
}

void ImagingSectionEnter(ImagingSectionCookie *cookie) { (void)cookie; }
void ImagingSectionLeave(ImagingSectionCookie *cookie) { (void)cookie; }
void *ImagingError_MemoryError(void) { die("out of memory"); return NULL; }
void *ImagingError_ModeError(void) { die("unsupported mode"); return NULL; }
void *ImagingError_ValueError(const char *message) { die(message); return NULL; }

int main(int argc, char **argv) {
    if (argc != 7) die("usage: oracle INPUT OUTPUT WIDTH HEIGHT TARGET_WIDTH TARGET_HEIGHT");
    int width = atoi(argv[3]), height = atoi(argv[4]);
    Imaging input = ImagingNewDirty(IMAGING_MODE_RGB, width, height);
    FILE *file = fopen(argv[1], "rb");
    if (!file) die("open input");
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (fread(input->image[y] + x * 4, 1, 3, file) != 3) die("read input");
        }
    }
    if (fclose(file)) die("close input");
    float box[4] = {0, 0, (float)width, (float)height};
    Imaging output = ImagingResample(input, atoi(argv[5]), atoi(argv[6]), IMAGING_TRANSFORM_BICUBIC, box);
    if (!output) die("resize");
    file = fopen(argv[2], "wb");
    if (!file) die("open output");
    for (int y = 0; y < output->ysize; ++y) {
        for (int x = 0; x < output->xsize; ++x) {
            if (fwrite(output->image[y] + x * 4, 1, 3, file) != 3) die("write output");
        }
    }
    if (fclose(file)) die("close output");
    ImagingDelete(output);
    ImagingDelete(input);
    return 0;
}
