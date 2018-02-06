#include "AveragePoolingLayer.h"

AveragePoolingLayer::AveragePoolingLayer(std::string name, std::string bottomBlobName, std::string topBlobName)
{
    this->name = name;

    bottomBlobNames = std::vector<std::string>();
    topBlobs        = std::vector<Blob *>();

    bottomBlobNames.push_back(bottomBlobName);
    topBlobs.push_back(new Blob(topBlobName));
}

void AveragePoolingLayer::setUp()
{
    std::cout << "Setting up average pooling layer..." << std::endl;

    outputWidth  = ((bottomBlobs[0]->width  - kernelSize) / stride) + 1;
    outputHeight = ((bottomBlobs[0]->height - kernelSize) / stride) + 1;

    std::cout << "\t" << bottomBlobs[0]->channels << "x" << outputHeight << "x" << outputWidth << std::endl;

    diffMatrix = MatrixXf(kernelSize, kernelSize); //TODO if training only

    topBlobs[0]->reshape(bottomBlobs[0]->channels, outputHeight, outputWidth);
}

void AveragePoolingLayer::forward()
{
    topBlobs[0]->dataMatrix.resize(outputHeight, (outputWidth * bottomBlobs[0]->channels));

    for(int channel = 0; channel < bottomBlobs[0]->channels; channel++)
    {
        int depth = (channel * bottomBlobs[0]->width);

        for(int c = 0; c < outputWidth; c++)
        {
            int col = (c * stride);

            for(int r = 0; r < outputHeight; r++)
            {
                int row = (r * stride);

                MatrixXf block = bottomBlobs[0]->dataMatrix.block(row, col + depth, kernelSize, kernelSize).matrix();

                float average = block.sum() / block.size();

                topBlobs[0]->dataMatrix.block(r, c + (channel * outputWidth), 1, 1).fill(average);
            }
        }
    }
}

void AveragePoolingLayer::backward()
{
    topBlobs[0]->diffMatrix.resize(outputHeight, (outputWidth * bottomBlobs[0]->channels));
    bottomBlobs[0]->diffMatrix.resize(bottomBlobs[0]->height, (bottomBlobs[0]->width * bottomBlobs[0]->channels));

    for(int channel = 0; channel < bottomBlobs[0]->channels; channel++)
    {
        int depth = (channel * bottomBlobs[0]->width);

        for(int c = 0; c < outputWidth; c++)
        {
            int col = (c * stride);

            for(int r = 0; r < outputHeight; r++)
            {
                int row = (r * stride);

                float diff = topBlobs[0]->diffMatrix.block(r, c + (channel * outputWidth), 1, 1).maxCoeff();

                diffMatrix.fill(diff / (kernelSize * kernelSize));

                bottomBlobs[0]->diffMatrix.block(row, col + depth, kernelSize, kernelSize) = diffMatrix;
            }
        }
    }
}

AveragePoolingLayer* AveragePoolingLayer::setKernelSize(int kernelSize)
{
    this->kernelSize = kernelSize;

    return this;
}

AveragePoolingLayer* AveragePoolingLayer::setStride(int stride)
{
    this->stride = stride;

    return this;
}
