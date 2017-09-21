#include <cstdlib>
#include <cmath>
#include "MathUtils.h"

//Box Muller method
float MathUtils::randomGaussian(float mean, float std)
{
    static float n2       = 0.0f;
    static int   n2Cached = 0;

    if(!n2Cached)
    {
        float x, y, r;

        do
        {
            x = 2.0f * rand()/RAND_MAX - 1.0f;
            y = 2.0f * rand()/RAND_MAX - 1.0f;

            r = (x * x) + (y * y);
        }
        while((r == 0.0f) || (r > 1.0f));
        {
            float d  = sqrtf(-2.0f * logf(r) / r);
            float n1 = x * d;

            n2 = y * d;

            float result = n1*std + mean;

            n2Cached = 1;

            return result;
        }
    }
    else
    {
        n2Cached = 0;

        return (n2 * std) + mean;
    }
}