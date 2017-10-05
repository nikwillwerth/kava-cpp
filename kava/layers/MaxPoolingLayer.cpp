#include "MaxPoolingLayer.h"

MaxPoolingLayer::MaxPoolingLayer(std::string name, std::string bottomBlobName, std::string topBlobName)
{
    this->name = name;

    bottomBlobNames = std::vector<std::string>();
    topBlobs        = std::vector<Blob *>();

    bottomBlobNames.push_back(bottomBlobName);
    topBlobs.push_back(new Blob(topBlobName));
}

void MaxPoolingLayer::setUp()
{
    std::cout << "Setting up max pooling layer..." << std::endl;

    outputWidth  = ((bottomBlobs[0]->width  - kernelSize) / stride) + 1;
    outputHeight = ((bottomBlobs[0]->height - kernelSize) / stride) + 1;

    std::cout << "\t" << bottomBlobs[0]->channels << "x" << outputHeight << "x" << outputWidth << std::endl;

    maxIndices = MatrixXf(bottomBlobs[0]->height, (bottomBlobs[0]->width * bottomBlobs[0]->channels));

    topBlobs[0]->reshape(bottomBlobs[0]->channels, outputHeight, outputWidth);
}

void MaxPoolingLayer::forward()
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

                float maxValue = -1e10;
                float maxIndex = -1;

                for(int i = 0; i < block.size(); i++)
                {
                    float value = block.data()[i];

                    if(value > maxValue)
                    {
                        maxValue = value;
                        maxIndex = i;
                    }
                }

                maxIndices.block(r, c + (channel * outputWidth), 1, 1).fill(maxIndex);

                topBlobs[0]->dataMatrix.block(r, c + (channel * outputWidth), 1, 1).fill(maxValue);
            }
        }
    }
}

void MaxPoolingLayer::backward()
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

                int   maxIndex = (int)maxIndices.block(r, c + (channel * outputWidth), 1, 1).maxCoeff();
                float diff     = topBlobs[0]->diffMatrix.block(r, c + (channel * outputWidth), 1, 1).maxCoeff();

                float *diffArray = new float[kernelSize * kernelSize];

                for(int i = 0; i < (kernelSize * kernelSize); i++)
                {
                    if(i != maxIndex)
                    {
                        diffArray[i] = 0;
                    }
                    else
                    {
                        diffArray[i] = diff;
                    }
                }

                MatrixXf diffMatrix = MatrixXf(kernelSize, kernelSize);

                new (&diffMatrix) Map<MatrixXf>(diffArray, kernelSize, kernelSize);

                bottomBlobs[0]->diffMatrix.block(row, col + depth, kernelSize, kernelSize) = diffMatrix;
            }
        }
    }
}

MaxPoolingLayer* MaxPoolingLayer::setKernelSize(int kernelSize)
{
    this->kernelSize = kernelSize;

    return this;
}

MaxPoolingLayer* MaxPoolingLayer::setStride(int stride)
{
    this->stride = stride;

    return this;
}