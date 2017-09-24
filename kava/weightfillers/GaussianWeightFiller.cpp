#include "GaussianWeightFiller.h"
#include "../utils/RandomUtils.h"

void GaussianWeightFiller::fill(Blob *blob, int numInputs, int numOutputs)
{
    for(int i = 0; i < blob->count; i++)
    {
        blob->data[i] = RandomUtils::getRandomGaussian(0, 1);
    }

    blob->putDataIntoMatrix();
}