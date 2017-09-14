#ifndef KAVA_CPP_EUCLIDEANLOSSLAYER_H
#define KAVA_CPP_EUCLIDEANLOSSLAYER_H

#include "Layer.h"

class EuclideanLossLayer : public Layer
{
public:
    EuclideanLossLayer(std::string name, std::string bottomBlobOneName, std::string bottomBlobTwoName, std::string topBlobName);

    void setUp();
    void forward();
    void backward();

    MatrixXf diffMatrix;
};

#endif //KAVA_CPP_EUCLIDEANLOSSLAYER_H