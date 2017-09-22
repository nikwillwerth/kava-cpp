#include <iostream>
#include "InnerProductLayer.h"

InnerProductLayer::InnerProductLayer(const std::string name, std::string bottomBlobName, std::string topBlobName, int numOutputs)
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

void InnerProductLayer::setUp()
{
    std::cout << "Setting up inner product layer..." << std::endl;

    numInputs = bottomBlobs[0]->count;

    weightBlobs[0]->reshape(1, numOutputs, numInputs);
    topBlobs[0]->reshape(   1, 1,          numOutputs);

    WeightFiller::getWeightFillerWithType(weightFillerType)->fill(weightBlobs[0], numOutputs, numInputs);
}

void InnerProductLayer::forward()
{
    bottomBlobs[0]->dataMatrix.resize(numInputs, 1);

    topBlobs[0]->dataMatrix.noalias() = weightBlobs[0]->dataMatrix * bottomBlobs[0]->dataMatrix;
}

void InnerProductLayer::backward()
{
    topBlobs[0]->diffMatrix.resize(numOutputs, 1);

    bottomBlobs[0]->diffMatrix.noalias() = weightBlobs[0]->dataMatrix.transpose() * topBlobs[0]->diffMatrix;
    weightBlobs[0]->diffMatrix.noalias() = topBlobs[0]->diffMatrix * bottomBlobs[0]->dataMatrix.transpose();
}