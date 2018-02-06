#include "ConvolutionalLayer.h"

ConvolutionalLayer::ConvolutionalLayer(const std::string name, std::string bottomBlobName, std::string topBlobName, int numOutputs)
{
    this->name       = name;
    this->numOutputs = numOutputs;

    bottomBlobNames = std::vector<std::string>();
    topBlobs        = std::vector<Blob *>();
    weightBlobs     = std::vector<Blob *>();

    bottomBlobNames.push_back(bottomBlobName);
    topBlobs.push_back(   new Blob(topBlobName));
    weightBlobs.push_back(new Blob("weights"));
}

void ConvolutionalLayer::setUp()
{
    std::cout << "Setting up convolutional layer..." << std::endl;

    outputWidth  = ((bottomBlobs[0]->width  - kernelSize + (2 * padding)) / stride) + 1;
    outputHeight = ((bottomBlobs[0]->height - kernelSize + (2 * padding)) / stride) + 1;

    std::cout << "\t" << numOutputs << "x" << outputHeight << "x" << outputWidth << std::endl;

    kernelArea   = kernelSize  * kernelSize;
    kernelLength = kernelArea  * bottomBlobs[0]->channels;
    outputLength = outputWidth * outputHeight;

    im2colMatrix = MatrixXf(outputLength, kernelLength);

    topBlobs[0]->reshape(numOutputs, outputHeight, outputWidth);
    weightBlobs[0]->reshape(1, kernelLength, numOutputs);

    int fanIn  = (bottomBlobs[0]->channels * kernelSize * kernelSize);
    int fanOut = (numOutputs * kernelSize * kernelSize);

    WeightFiller::getWeightFillerWithType(weightFillerType)->fill(weightBlobs[0], fanIn, fanOut);
}

void ConvolutionalLayer::forward()
{
    //im2col(I)
    int rowIndex = 0;

    for(int c = 0; c < outputWidth; c++)
    {
        int col = (c * stride);

        for(int r = 0; r < outputHeight; r++)
        {
            int row = (r * stride);

            for(int channel = 0; channel < bottomBlobs[0]->channels; channel++)
            {
                int colIndex = (col + (channel * bottomBlobs[0]->width));

                MatrixXf block = bottomBlobs[0]->dataMatrix.block(row, colIndex, kernelSize, kernelSize).matrix();
                block.resize(1, kernelArea);

                int startColumn = channel * kernelArea;

                im2colMatrix.block(rowIndex, startColumn, 1, kernelArea) = block;
            }

            rowIndex++;
        }
    }

    //im2col(I) * W
    MatrixXf result = im2colMatrix * weightBlobs[0]->dataMatrix;

    topBlobs[0]->dataMatrix.resize(outputHeight, outputWidth * numOutputs);

    for(int i = 0; i < numOutputs; i++)
    {
        MatrixXf block = result.col(i).matrix();
        block.resize(outputHeight, outputWidth);

        topBlobs[0]->dataMatrix.block(0, i * outputWidth, outputHeight, outputWidth) = block;
    }
}

void ConvolutionalLayer::backward()
{
    topBlobs[0]->diffMatrix.resize(outputHeight, outputWidth * numOutputs);

    MatrixXf dY = MatrixXf(outputLength, numOutputs);

    for(int i = 0; i < numOutputs; i++)
    {
        MatrixXf block = topBlobs[0]->diffMatrix.block(0, i * outputWidth, outputHeight, outputWidth).matrix();
        block.resize(outputLength, 1);
        dY.block(0, i, outputLength, 1) = block;
    }

    bottomBlobs[0]->diffMatrix.noalias() = dY * weightBlobs[0]->dataMatrix.transpose();
    weightBlobs[0]->diffMatrix.noalias() = im2colMatrix.transpose() * dY;
}

ConvolutionalLayer* ConvolutionalLayer::setKernelSize(int kernelSize)
{
    this->kernelSize = kernelSize;

    return this;
}

ConvolutionalLayer* ConvolutionalLayer::setStride(int stride)
{
    this->stride = stride;

    return this;
}

ConvolutionalLayer* ConvolutionalLayer::setPadding(int padding)
{
    this->padding = padding;

    return this;
}