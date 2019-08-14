#include <math.h>
#include <stddef.h>

inline float calc_dist(const float * restrict a, const float * restrict b, size_t n, float param) {
    float sum = 0.;
    for(size_t i=0; i<n; i++) sum += (a[i]-b[i])*(a[i]-b[i]);
    return exp(param*sqrt(sum)) - 1.;
}

void dijkstra_impl(size_t start, size_t stop, size_t T, size_t n, const float * restrict x, float * restrict dist, char * restrict visited, int * restrict pred, float param) {
    for(size_t i = 0; i < T; i++) { pred[i] = -1; dist[i]=INFINITY; visited[i]=0; }
    dist[start] = 0.;

    for(size_t count=T; count > 0; count--) {
        float mindist = INFINITY;
        size_t u = start;
        for(size_t i=0; i<T; i++) { if(!visited[i] && dist[i] < mindist) {mindist=dist[i]; u=i;} }
        visited[u] = 1;

        if(u==stop) return;

        for(size_t v=0; v<T; v++) {
            if(!visited[v]) {
                float d = calc_dist(&x[u*n], &x[v*n], n, param);
                if(d + dist[u] < dist[v]) {
                    dist[v] = d + dist[u];
                    pred[v] = u;
                 }
             }
        }
    }
}