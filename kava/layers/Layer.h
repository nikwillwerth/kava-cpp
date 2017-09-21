#ifndef KAVA_LAYER_H
#define KAVA_LAYER_H

#include "../Blob.h"
#include "../weightfillers/WeightFiller.h"
#include <string>
#include <vector>

class Layer
{
public:
    virtual void setUp()    {}
    virtual void forward()  {}
    virtual void backward() {}

    Layer* setWeightFiller(WeightFiller::Type type);

    std::string         name;
    std::vector<std::string> bottomBlobNames;
    std::vector<Blob *> bottomBlobs;
    std::vector<Blob *> topBlobs;
    std::vector<Blob *> weightBlobs;

    bool isInPlace = false;

protected:
    WeightFiller::Type weightFillerType = WeightFiller::Type::Gaussian;
};

#endif //KAVA_LAYER_H