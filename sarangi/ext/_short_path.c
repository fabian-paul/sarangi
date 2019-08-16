#include <math.h>
#include <stddef.h>
#include <signal.h>

static volatile sig_atomic_t interrupted;
static void (*old_handler)(int);

static void signal_handler(int signo) {
  interrupted = 1;
}

static void sigint_on(void) {
  interrupted = 0;
  old_handler = signal(SIGINT, signal_handler);
}

static void sigint_off(void) {
  if(old_handler != SIG_ERR) {
    signal(SIGINT, old_handler);
    if(interrupted) raise(SIGINT);
  }
}


inline double calc_dist(const float * restrict a, const float * restrict b, size_t n, float param) {
    float sum = 0.;
    for(size_t i=0; i<n; i++) sum += (a[i]-b[i])*(a[i]-b[i]);
    /* return exp((double)(param*sqrt(sum))) - 1.; */
    return expm1((double)(param*sqrt(sum)));
}

inline double calc_log_dist(const float * restrict a, const float * restrict b, size_t n, float param) {
    float sum = 0.;
    for(size_t i=0; i<n; i++) sum += (a[i]-b[i])*(a[i]-b[i]);
    double c = param*sqrt(sum);
    //return c + log(-expm1(-c));
    return c + log1p(-exp(-c));
}

inline double log_add_exp(double a, double b) {
    //double c = fmaxf(a, b);
    //return c + log(exp(a - c) + exp(b - c));
    if (a > b)
        return a + log1p(exp(b - a));
    else
        return b + log1p(exp(a - b));
}

void dijkstra_impl(size_t start, size_t stop, size_t T, size_t n, const float * restrict x, double * restrict dist, char * restrict visited, int * restrict pred, float param) {
    sigint_on();
    for(size_t i = 0; i < T; i++) { pred[i] = -1; dist[i]=INFINITY; visited[i]=0; }
    dist[start] = 0.;

    for(size_t count=T; count > 0 && !interrupted; count--) {
        double mindist = INFINITY;
        size_t u = start;
        for(size_t i=0; i<T; i++) { if(!visited[i] && dist[i] < mindist) {mindist=dist[i]; u=i;} }
        visited[u] = 1;

        if(u==stop) {sigint_off(); return;}

        for(size_t v=0; v<T; v++) {
            if(!visited[v]) {
                /*double d = calc_dist(&x[u*n], &x[v*n], n, param);*/
                double d = calc_log_dist(&x[u*n], &x[v*n], n, param);
                if(d + dist[u] < dist[v]) {
                    /*dist[v] = d + dist[u];*/
                    dist[v] = log_add_exp(d, dist[u]);
                    pred[v] = u;
                 }
             }
        }
    }
    sigint_off();
}
