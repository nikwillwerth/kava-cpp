#include "MSRAWeightFiller.h"
#include "../utils/MathUtils.h"

void MSRAWeightFiller::fill(Blob *blob, int numInputs, int numOutputs)
{
    float var = (2.0f / (numInputs * numOutputs));
    float std = sqrtf(var);

    for(int i = 0; i < blob->count; i++)
    {
        blob->data[i] = MathUtils::randomGaussian(0, std);
    }

    blob->putDataIntoMatrix();
}