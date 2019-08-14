#ifndef SHORT_PATH_H
#define SHORT_PATH_H
#include <stddef.h>

void dijkstra_impl(size_t start, size_t stop, size_t T, size_t n, const float * restrict x, float * restrict dist, char * restrict visited, int * restrict pred, float param);

#endif