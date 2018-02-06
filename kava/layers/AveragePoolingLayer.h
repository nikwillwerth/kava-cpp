#ifndef KAVA_CPP_AVERAGEPOOLINGLAYER_H
#define KAVA_CPP_AVERAGEPOOLINGLAYER_H

#include "Layer.h"

class AveragePoolingLayer : public Layer
{
public:
    AveragePoolingLayer(std::string name, std::string bottomBlobName, std::string topBlobName);

    void setUp();
    void forward();
    void backward();

    AveragePoolingLayer* setKernelSize(int kernelSize);
    AveragePoolingLayer* setStride(int stride);

    int kernelSize = 2;
    int stride     = 2;
    int outputWidth;
    int outputHeight;

private:
    MatrixXf diffMatrix;
};

#endif //KAVA_CPP_AVERAGEPOOLINGLAYER_H
