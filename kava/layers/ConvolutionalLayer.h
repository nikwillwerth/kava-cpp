#ifndef KAVA_CPP_CONVOLUTIONALLAYER_H
#define KAVA_CPP_CONVOLUTIONALLAYER_H

#include <string>
#include "../Eigen/Dense"
#include "Layer.h"

using namespace Eigen;

class ConvolutionalLayer : public Layer
{
public:
    ConvolutionalLayer(std::string name, std::string bottomBlobName, std::string topBlobName, int numOutputs);

    void setUp();
    void forward();
    void backward();

private:
    int numOutputs;
    int kernelSize = 3;
    int stride     = 1;
    int padding    = 0;
    int kernelLength;
    int outputWidth;
    int outputHeight;
    int outputLength;

    MatrixXf im2colMatrix;
};

#endif //KAVA_CPP_CONVOLUTIONALLAYER_H