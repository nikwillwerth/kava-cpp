#ifndef KAVA_CPP_XAVIERWEIGHTFILLER_H
#define KAVA_CPP_XAVIERWEIGHTFILLER_H

#include "WeightFiller.h"

class XavierWeightFiller : public WeightFiller
{
    void fill(Blob *blob, int numInputs, int numOutputs);
};

#endif //KAVA_CPP_XAVIERWEIGHTFILLER_H