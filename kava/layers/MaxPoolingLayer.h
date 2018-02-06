#ifndef KAVA_CPP_MAXPOOLINGLAYER_H
#define KAVA_CPP_MAXPOOLINGLAYER_H

#include "Layer.h"

class MaxPoolingLayer : public Layer
{
public:
    MaxPoolingLayer(std::string name, std::string bottomBlobName, std::string topBlobName);

    void setUp();
    void forward();
    void backward();

    MaxPoolingLayer* setKernelSize(int kernelSize);
    MaxPoolingLayer* setStride(int stride);

    int kernelSize = 2;
    int stride     = 2;
    int outputWidth;
    int outputHeight;

private:
    MatrixXf maxIndices;
    MatrixXf diffMatrix;
};

#endif //KAVA_CPP_MAXPOOLINGLAYER_H
