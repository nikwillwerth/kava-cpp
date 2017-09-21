#ifndef KAVA_CPP_WEIGHTFILLER_H
#define KAVA_CPP_WEIGHTFILLER_H

#include <iostream>
#include "../Blob.h"

class WeightFiller
{
public:
    enum Type
    {
        Constant,
        Gaussian,
        MSRA,
        Xavier
    };

    virtual void fill(Blob *blob, int numInputs, int numOutputs)
    {
        std::cout << "Wrong" << std::endl;
    }

    static WeightFiller* getWeightFillerWithType(Type type);
};

#endif //KAVA_CPP_WEIGHTFILLER_H