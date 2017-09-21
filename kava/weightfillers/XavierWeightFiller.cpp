#include "XavierWeightFiller.h"
#include "../utils/MathUtils.h"

void XavierWeightFiller::fill(Blob *blob, int numInputs, int numOutputs)
{
    float var = (1.0f / ((numInputs + numOutputs) / 2.0f));
    float std = sqrtf(var);

    for(int i = 0; i < blob->count; i++)
    {
        blob->data[i] = MathUtils::randomGaussian(0, std);
    }

    new (&blob->dataMatrix) Map<MatrixXf>(blob->data, blob->height, blob->width);
}