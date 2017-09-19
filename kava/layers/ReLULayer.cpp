//
// Created by nik on 9/18/17.
//

#include <iostream>
#include "ReLULayer.h"

ReLULayer::ReLULayer(const std::string name, std::string bottomBlobName, std::string topBlobName)
{
    this->name = name;

    bottomBlobNames = std::vector<std::string>();
    topBlobs        = std::vector<Blob *>();

    bottomBlobNames.push_back(bottomBlobName);
    topBlobs.push_back(   new Blob(topBlobName));
}

void ReLULayer::setUp()
{
    std::cout << "Setting up ReLU layer..." << std::endl;

    topBlobs[0]->reshape(bottomBlobs[0]->channels, bottomBlobs[0]->height, bottomBlobs[0]->width);
}

void ReLULayer::forward()
{
    reluMask = (bottomBlobs[0]->dataMatrix.array() >= 0.0f).matrix().cast<float>();

    topBlobs[0]->dataMatrix.noalias() = bottomBlobs[0]->dataMatrix.cwiseProduct(reluMask);
}

void ReLULayer::backward()
{
    reluMask.resize(topBlobs[0]->diffMatrix.rows(), topBlobs[0]->diffMatrix.cols());

    bottomBlobs[0]->diffMatrix.noalias() = topBlobs[0]->diffMatrix.cwiseProduct(reluMask);
}