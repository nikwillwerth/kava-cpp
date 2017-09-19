//
#ifndef KAVA_CPP_SOFTMAXWITHLOSSLAYER_H
#define KAVA_CPP_SOFTMAXWITHLOSSLAYER_H

#include "Layer.h"

class SoftmaxWithLossLayer : public Layer
{
public:
    SoftmaxWithLossLayer(std::string name, std::string bottomBlobOneName, std::string bottomBlobTwoName, std::string topBlobName);

    void setUp();
    void forward();
    void backward();

    MatrixXf diffMatrix;
};

#endif //KAVA_CPP_SOFTMAXWITHLOSSLAYER_H