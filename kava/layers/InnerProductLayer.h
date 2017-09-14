#ifndef KAVA_INNERPRODUCTLAYER_H
#define KAVA_INNERPRODUCTLAYER_H

#include "Layer.h"

class InnerProductLayer : public Layer
{
public:
    InnerProductLayer(std::string name, std::string bottomBlobName, std::string topBlobName, int numOutputs);

    void setUp();
    void forward();
    void backward();

    int numInputs;
    int numOutputs;
};

#endif //KAVA_INNERPRODUCTLAYER_H