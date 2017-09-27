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

    kernelArea   = kernelSize  * kernelSize;
    kernelLength = kernelArea  * bottomBlobs[0]->channels;
    outputLength = outputWidth * outputHeight * bottomBlobs[0]->channels;

    im2colMatrix = MatrixXf(outputLength, kernelLength);

    topBlobs[0]->reshape(numOutputs, outputHeight, outputWidth);
    weightBlobs[0]->reshape(1, kernelLength, numOutputs);

    WeightFiller::getWeightFillerWithType(weightFillerType)->fill(weightBlobs[0], kernelLength, numOutputs);
}

void ConvolutionalLayer::forward()
{
    //im2col(I)
    int rowIndex = 0;

    for(int r = 0; r < outputHeight; r++)
    {
        int row = (r * stride);

        for(int c = 0; c < outputWidth; c++)
        {
            int col = (c * stride);

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
    topBlobs[0]->dataMatrix = im2colMatrix * weightBlobs[0]->dataMatrix;
    topBlobs[0]->dataMatrix.resize(outputHeight, outputWidth * numOutputs);
}

void ConvolutionalLayer::backward()
{
    topBlobs[0]->diffMatrix.resize(outputWidth * outputHeight, numOutputs);

    //std::cout << im2colMatrix.rows() << "x" << im2colMatrix.cols() << std::endl;
    //std::cout << topBlobs[0]->diffMatrix.rows() << "x" << topBlobs[0]->diffMatrix.cols() << std::endl;

    //dW
    weightBlobs[0]->diffMatrix = im2colMatrix.transpose() * topBlobs[0]->diffMatrix;
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