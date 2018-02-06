#include "XavierWeightFiller.h"
#include "../utils/RandomUtils.h"

void XavierWeightFiller::fill(Blob *blob, int numInputs, int numOutputs)
{
    float std = sqrtf(3.0f / numInputs);

    for(int i = 0; i < blob->count; i++)
    {
        blob->data[i] = RandomUtils::getRandomUniform(0, std);
    }

    blob->putDataIntoMatrix();
}