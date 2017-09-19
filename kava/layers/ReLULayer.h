#ifndef KAVA_CPP_RELULAYER_H
#define KAVA_CPP_RELULAYER_H

#include <string>
#include "Layer.h"

class ReLULayer : public Layer
{
public:
    ReLULayer(std::string name, std::string bottomBlobName, std::string topBlobName);

    void setUp();
    void forward();
    void backward();

private:
    MatrixXf reluMask;
};

#endif //KAVA_CPP_RELULAYER_H