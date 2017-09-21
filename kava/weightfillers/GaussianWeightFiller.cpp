#include "GaussianWeightFiller.h"
#include "../utils/MathUtils.h"

void GaussianWeightFiller::fill(Blob *blob, int numInputs, int numOutputs)
{
    float var = 1.0f;
    float std = sqrtf(var);

    for(int i = 0; i < blob->count; i++)
    {
        blob->data[i] = MathUtils::randomGaussian(0, std);
    }

    new (&blob->dataMatrix) Map<MatrixXf>(blob->data, blob->height, blob->width);
}