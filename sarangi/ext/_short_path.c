#include <math.h>
#include <stddef.h>
#include <signal.h>
#include <assert.h>

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

inline double log1mexp(double x) {
    /* Maechler, M. Accurately Computing log(1 - exp(- |a|)) Assessed by the Rmpfr package Cran, The Comprehensive R Archive Network. */
    if (x <= 0.6931471805599453)
        return log(-expm1(-x));
    else
        return log1p(-exp(-x));
}

inline double calc_log_dist(const float * restrict a, const float * restrict b, size_t n, float param) {
    float sum = 0.;
    for(size_t i=0; i<n; i++) sum += (a[i]-b[i])*(a[i]-b[i]);
    double c = param*sqrt(sum);
    /* The return value is log(exp(param*distance) - 1) where the purpose of the logarithm is to avoid overflows and being able to compare large values */
    /* We use the identity log(exp(x) - 1) = x + log(1 - exp(-x)) and several high-precision implementations of special functions to avoid numerical errors. */
    //return c + log(-expm1(-c));
    //double r = c + log1p(-exp(-c));
    double r = c + log1mexp(c);
    assert(r >= 0);
    return r;
}

inline double log1pexp(double x) {
    /* Maechler, M. Accurately Computing log(1 - exp(- |a|)) Assessed by the Rmpfr package Cran, The Comprehensive R Archive Network. */
    if (x<=-37)
        return exp(x);
    else if (x<18)
        return log1p(exp(x));
    else if (x<33.3)
        return x + exp(-x);
    else
        return x;
}

inline double log_add_exp(double a, double b) {
    //double c = fmaxf(a, b);
    //return c + log(exp(a - c) + exp(b - c));
    if (a > b)
        return a + log1pexp(b - a); // log1p(exp(b - a));
    else
        return b + log1pexp(a - b); // log1p(exp(a - b));
}

void dijkstra_impl(size_t start, size_t stop, size_t T, size_t n, const float * restrict x, double * restrict dist, char * restrict visited, int * restrict pred, float param, int logspace) {
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
                double q;
                /*double d = calc_dist(&x[u*n], &x[v*n], n, param);*/
                if (logspace) {
                    double d = calc_log_dist(&x[u*n], &x[v*n], n, param);
                    q = log_add_exp(d, dist[u]);
                } else {
                    double d = calc_dist(&x[u*n], &x[v*n], n, param);
                    q = d + dist[u];
                };
                if(q < dist[v]) {
                    /*dist[v] = d + dist[u];*/
                    dist[v] = q;
                    pred[v] = u;
                 }
             }
        }
    }
    sigint_off();
}
