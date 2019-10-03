#include <math.h>
#include <stdlib.h>

static float bistable_potential(float x, float y)
{
    return -exp(-(x-1)*(x-1) - y*y) - exp(-(x+1)*(x+1) - y*y) \
           + 5*exp(-0.32*(x*x + y*y + 20*(x + y)*(x + y))) \
           + 32/1875*(x*x*x*x + y*y*y*y) + 2/15*exp(-2-4*y);
}


void propagate_bistable_(float *x0, float *y0, int n_steps, unsigned short rng_seed)
{
    float e;
    float x = *x0;
    float y = *y0;
    unsigned short rngs[3];
    rngs[0]= 0x330E;
    rngs[1]= 666;
    rngs[2]= rng_seed;

    e = bistable_potential(x, y);
    for(int t=0; t<n_steps; t++) {
        float x_prime = x + (erand48(rngs) - .5)*0.05;
        float y_prime = y + (erand48(rngs) - .5)*0.05;
        float e_prime = bistable_potential(x_prime, y_prime);
        if (exp(e - e_prime) > erand48(rngs)) {
            x = x_prime;
            y = y_prime;
            e = e_prime;
        }
    }
    *x0 = x;
    *y0 = y;
}