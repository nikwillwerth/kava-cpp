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

    float variance = (2.0f / (numInputs + numOutputs));

    for(int i = 0; i < weightBlobs[0]->count; i++)
    {
        float randy = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

        weightBlobs[0]->data[i] = ((randy * 2.0f) - 1.0f) * variance;
    }

    new (&weightBlobs[0]->dataMatrix) Map<MatrixXf>(weightBlobs[0]->data, numOutputs, numInputs);
}

void InnerProductLayer::forward()
{
    bottomBlobs[0]->dataMatrix.resize(1,         numInputs);
    weightBlobs[0]->dataMatrix.resize(numInputs, numOutputs);

    topBlobs[0]->dataMatrix.noalias() = bottomBlobs[0]->dataMatrix * weightBlobs[0]->dataMatrix;
}

void InnerProductLayer::backward()
{
    topBlobs[0]->diffMatrix.resize(topBlobs[0]->count / numOutputs, numOutputs);
    weightBlobs[0]->dataMatrix.resize(numOutputs, numInputs);

    bottomBlobs[0]->diffMatrix.noalias() = topBlobs[0]->diffMatrix * weightBlobs[0]->dataMatrix;

    topBlobs[0]->diffMatrix.resize(topBlobs[0]->diffMatrix.cols(), topBlobs[0]->diffMatrix.rows());
    bottomBlobs[0]->dataMatrix.resize(1, numInputs);

    weightBlobs[0]->diffMatrix.noalias() = topBlobs[0]->diffMatrix * bottomBlobs[0]->dataMatrix;
}