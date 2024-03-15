// Costin Didoaca 
// 333CA
// Author: APD team, except where source was noted

#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
//#include "pthread_barrier_mac.h"


#define CONTOUR_CONFIG_COUNT    16
#define FILENAME_MAX_SIZE       50
#define STEP                    8
#define SIGMA                   200
#define RESCALE_X               2048
#define RESCALE_Y               2048

#define CLAMP(v, min, max) if(v < min) { v = min; } else if(v > max) { v = max; }

// thread routine
typedef struct {
    int thread_id;
    int size;
    unsigned char **grid;
    ppm_image *img_original;
    ppm_image *img_rescaled;
    ppm_image **contour_map;
    pthread_barrier_t *barrier;
} ppm_thread_routine;

void rescale_image_if_needed(ppm_thread_routine *info);
void sample_image_grid(ppm_thread_routine *info);
void initialize_contour_map(ppm_thread_routine *info);
void apply_contour_to_image(ppm_thread_routine *info);
ppm_image *allocate_rescaled_image(ppm_image *original);

// Updates a particular section of an image with the corresponding contour pixels.
// Used to create the complete contour image.
void update_image(ppm_image *image, ppm_image *contour, int x, int y) {
    for (int i = 0; i < contour->x; i++) {
        for (int j = 0; j < contour->y; j++) {
            int contour_pixel_index = contour->x * i + j;
            int image_pixel_index = (x + i) * image->y + y + j;

            image->data[image_pixel_index].red = contour->data[contour_pixel_index].red;
            image->data[image_pixel_index].green = contour->data[contour_pixel_index].green;
            image->data[image_pixel_index].blue = contour->data[contour_pixel_index].blue;
        }
    }
}


// Calls `free` method on the utilized resources.
void free_resources(ppm_image *image, ppm_image **contour_map, unsigned char **grid, int step_x) {
    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        free(contour_map[i]->data);
        free(contour_map[i]);
    }
    free(contour_map);

    for (int i = 0; i <= image->x / step_x; i++) {
        free(grid[i]);
    }
    free(grid);

    free(image->data);
    free(image);
}

void *thread_routine(void *arg) {
    ppm_thread_routine *info = (ppm_thread_routine *) arg;

    rescale_image_if_needed(info);
    pthread_barrier_wait(info->barrier);

    sample_image_grid(info);
    pthread_barrier_wait(info->barrier);

    initialize_contour_map(info);
    pthread_barrier_wait(info->barrier);

    apply_contour_to_image(info);

    pthread_exit(NULL);
}

void rescale_image_if_needed(ppm_thread_routine *info) {
    // rescale logic implementation with threads    
    int nr_threads = info->size;
    //rescale
    if (info->img_original != info->img_rescaled) {
        int start_rescale = info->thread_id * info->img_rescaled->y / nr_threads;
        int end_rescale = fmin((info->thread_id + 1) * info->img_rescaled->y / nr_threads, info->img_rescaled->y);
        uint8_t sample[3];

         // use bicubic interpolation for scaling
        for (int i = 0; i < info->img_rescaled->x; i++) {
            for (int j = start_rescale; j < end_rescale; j++) {
                float u = (float)i / (float)(info->img_rescaled->x - 1);
                float v = (float)j / (float)(info->img_rescaled->y - 1);
                sample_bicubic(info->img_original, u, v, sample);

                info->img_rescaled->data[i * info->img_rescaled->y + j].red = sample[0];
                info->img_rescaled->data[i * info->img_rescaled->y + j].green = sample[1];
                info->img_rescaled->data[i * info->img_rescaled->y + j].blue = sample[2];
            }
        }
    }
}

void sample_image_grid(ppm_thread_routine *info) {
    // create and fullfil the grid
    
    int nr_threads = info->size;
    unsigned char **grid = info->grid;
    
    int p = info->img_rescaled->x / STEP;
    int q = info->img_rescaled->y / STEP;

    //start end points for sample_grid
    int start1 = info->thread_id * q / nr_threads;
    int end1 = fmin((info->thread_id + 1) * p / nr_threads, p);

    // Corresponds to step 1 of the marching squares algorithm, which focuses on sampling the image.
    // Builds a p x q grid of points with values which can be either 0 or 1, depending on how the
    // pixel values compare to the `sigma` reference value. The points are taken at equal distances
    // in the original image, based on the `step_x` and `step_y` arguments.
    for (int i = 0; i < p; i++) {
        for (int j = start1; j < end1; j++) {
            ppm_pixel curr_pixel = info->img_rescaled->data[i * STEP * info->img_rescaled->y + j * STEP];

            unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

            if (curr_color > SIGMA) {
                grid[i][j] = 0;
            } else {
                grid[i][j] = 1;
            }
        }
    }
    grid[p][q] = 0;

    int start2 = info->thread_id * p / nr_threads;
    int end2 = fmin((info->thread_id + 1) * p / nr_threads, p);

    // last sample points have no neighbors below / to the right, so we use pixels on the
    // last row / column of the input image for them
    for (int i = start2; i < end2; i++) {

        ppm_pixel curr_pixel = info->img_rescaled->data[i * STEP * info->img_rescaled->y + info->img_rescaled->x - 1];
        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > SIGMA) {
            grid[i][q] = 0;
        } else {
            grid[i][q] = 1;
        }
    }

    for (int j = start1; j < end1; j++) {
        ppm_pixel curr_pixel = info->img_rescaled->data[(info->img_rescaled->x - 1) * info->img_rescaled->y + j * STEP];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > SIGMA) {
            grid[p][j] = 0;
        } else {
            grid[p][j] = 1;
        }
    }
}

void initialize_contour_map(ppm_thread_routine *info) {
    // init contour map with threads (not needed)
    int nr_threads = info->size;

    int start_map = info->thread_id * CONTOUR_CONFIG_COUNT / nr_threads;
    int end_map = fmin((info->thread_id + 1) * CONTOUR_CONFIG_COUNT / nr_threads, CONTOUR_CONFIG_COUNT);

    // map init
    for (int i = start_map; i < end_map; i++) {
        char filename[FILENAME_MAX_SIZE];
        sprintf(filename, "./contours/%d.ppm", i);
        info->contour_map[i] = read_ppm(filename);
    }

}

void apply_contour_to_image(ppm_thread_routine *info) {
    // apply contour onto an image grid
    int nr_threads = info->size;
    unsigned char **grid = info->grid;

    int p = info->img_rescaled->x / STEP;
    int q = info->img_rescaled->y / STEP;

    int start = info->thread_id * q / nr_threads;
    int end = fmin((info->thread_id + 1) * p / nr_threads, p);
    
    for (int i = 0; i < p; i++) {
        for (int j = start; j < end; j++) {
            unsigned char k = 8 * grid[i][j] + 4 * grid[i][j + 1] + 2 * grid[i + 1][j + 1] + 1 * grid[i + 1][j];
            update_image(info->img_rescaled, info->contour_map[k], i * STEP, j * STEP);
        }
    }
}

ppm_image *allocate_rescaled_image(ppm_image *original) {
    // alloc rescaled image
    ppm_image *new_image = (ppm_image *)malloc(sizeof(ppm_image));
        if (!new_image) {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }

        new_image->x = RESCALE_X;
        new_image->y = RESCALE_Y;

        // alloc memory for pixel data
        new_image->data = (ppm_pixel *)malloc(new_image->x * new_image->y * sizeof(ppm_pixel));
        if (!new_image) {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }
        return new_image;
}

unsigned char **allocate_grid(ppm_image *img_tmp) {
    // alloc grid
    int p = img_tmp->x / STEP;
    int q = img_tmp->y / STEP;

    unsigned char **grid = (unsigned char **)malloc((p + 1) * sizeof(unsigned char*));
    if (!grid) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i <= p; i++) {
        grid[i] = (unsigned char *)malloc((q + 1) * sizeof(unsigned char));
        if (!grid[i]) {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }
    }
    return grid;
}

ppm_image **allocate_contour_map() {
    // alloc contour map
    ppm_image **map = (ppm_image **)malloc(CONTOUR_CONFIG_COUNT * sizeof(ppm_image *));
    if (!map) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }
    return map;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: ./tema1 <in_file> <out_file> <nr_threads>\n");
        return 1;
    }
    
    ppm_image *image = read_ppm(argv[1]);
    int nr_threads = atoi(argv[3]);
    pthread_t threads[nr_threads];
    pthread_barrier_t barrier;
    ppm_thread_routine thread_data[nr_threads];


    ppm_image *img_tmp = (image->x <= RESCALE_X && image->y <= RESCALE_Y) ? image : allocate_rescaled_image(image);
    unsigned char **grid = allocate_grid(img_tmp);
    ppm_image **map = allocate_contour_map();

    pthread_barrier_init(&barrier, NULL, nr_threads);

    // create threads-workers
    for (int i = 0; i < nr_threads; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].size = nr_threads;
        thread_data[i].img_rescaled = img_tmp;
        thread_data[i].img_original = image;
        thread_data[i].grid = grid;
        thread_data[i].barrier = &barrier;
        thread_data[i].contour_map = map;
        pthread_create(&threads[i], NULL, thread_routine, &thread_data[i]);
    }

    // wait for threads
    for(int i = 0; i < nr_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_barrier_destroy(&barrier);

    write_ppm(img_tmp, argv[2]);

    free_resources(img_tmp, map, grid, STEP);

    return 0;
}



