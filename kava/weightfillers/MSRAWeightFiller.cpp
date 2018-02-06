#include "MSRAWeightFiller.h"
#include "../utils/RandomUtils.h"

void MSRAWeightFiller::fill(Blob *blob, int numInputs, int numOutputs)
{
    float std = sqrtf(2.0f / numInputs);

    for(int i = 0; i < blob->count; i++)
    {
        blob->data[i] = RandomUtils::getRandomGaussian(0, std);
    }

    blob->putDataIntoMatrix();
}