#include <iostream>
#include "InnerProductLayer.h"

InnerProductLayer::InnerProductLayer(std::string name, std::string bottomBlobName, std::string topBlobName, int numOutputs)
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

    weightBlobs[0]->reshape(1, 1, numOutputs, numInputs);
    topBlobs[0]->reshape(   1, 1, 1,          numInputs);
}

void InnerProductLayer::forward()
{
    //std::cout << bottomBlobs[0]->dataMatrix->matrix() << std::endl;
    //std::cout << Eigen::Map<Eigen::RowVectorXf>(bottomBlobs[0]->data, bottomBlobs[0]->count) << std::endl;
    //std::cout << weightBlobs[0]->dataMatrix->matrix() << std::endl;

    //bottomBlobs[0]->dataMatrix->matrix() * weightBlobs[0]->dataMatrix->matrix();
}