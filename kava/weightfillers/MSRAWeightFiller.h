#ifndef KAVA_CPP_MSRAWEIGHTFILLER_H
#define KAVA_CPP_MSRAWEIGHTFILLER_H

#include "WeightFiller.h"

class MSRAWeightFiller : public WeightFiller
{
public:
    void fill(Blob *blob, int numInputs, int numOutputs);
};

#endif //KAVA_CPP_MSRAWEIGHTFILLER_H