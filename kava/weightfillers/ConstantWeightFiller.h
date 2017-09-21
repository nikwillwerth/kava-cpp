#ifndef KAVA_CPP_CONSTANTWEIGHTFILLER_H
#define KAVA_CPP_CONSTANTWEIGHTFILLER_H

#include "WeightFiller.h"

class ConstantWeightFiller : public WeightFiller
{
public:
    void fill(Blob *blob, int numInputs, int numOutputs);
};

#endif //KAVA_CPP_CONSTANTWEIGHTFILLER_H