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

    kernelLength = kernelSize  * kernelSize * bottomBlobs[0]->channels;
    outputLength = outputWidth * outputHeight;

    im2colMatrix = MatrixXf(outputLength, kernelLength);

    weightBlobs[0]->reshape(1, kernelLength, numOutputs);
    topBlobs[0]->reshape(   1, outputLength, numOutputs);

    WeightFiller::getWeightFillerWithType(weightFillerType)->fill(weightBlobs[0], kernelLength, numOutputs);

    std::cout << im2colMatrix << std::endl;

    im2colMatrix.block(0, 0, 3, 3) = bottomBlobs[0]->dataMatrix.block(0, 0, 3, 3);

    std::cout << im2colMatrix << std::endl;
}

void ConvolutionalLayer::forward()
{

}

void ConvolutionalLayer::backward()
{

}