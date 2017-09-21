#ifndef KAVA_CPP_GAUSSIANWEIGHTFILLER_H
#define KAVA_CPP_GAUSSIANWEIGHTFILLER_H

#include "WeightFiller.h"

class GaussianWeightFiller : public WeightFiller
{
    void fill(Blob *blob, int numInputs, int numOutputs);
};

#endif //KAVA_CPP_GAUSSIANWEIGHTFILLER_H