#include <iostream>
#include "EuclideanLossLayer.h"

EuclideanLossLayer::EuclideanLossLayer(std::string name, std::string bottomBlobOneName, std::string bottomBlobTwoName, std::string topBlobName)
{
    this->name = name;

    bottomBlobNames = std::vector<std::string>();
    topBlobs        = std::vector<Blob *>();

    bottomBlobNames.push_back(bottomBlobOneName);
    bottomBlobNames.push_back(bottomBlobTwoName);
    topBlobs.push_back(new Blob(topBlobName, 1, 1, 1));
}

void EuclideanLossLayer::setUp()
{
    std::cout << "Setting up euclidean loss layer..." << std::endl;

    if(bottomBlobs[0]->count != bottomBlobs[1]->count)
    {
        std::cout << "Blob sizes do not match in layer " << name << "(" << bottomBlobs[0]->count << " vs. " << bottomBlobs[1]->count << ")" << std::endl;
    }
}

void EuclideanLossLayer::forward()
{
    bottomBlobs[1]->dataMatrix.resize(bottomBlobs[0]->dataMatrix.rows(), bottomBlobs[0]->dataMatrix.cols());

    diffMatrix = bottomBlobs[0]->dataMatrix - bottomBlobs[1]->dataMatrix;

    float loss = (float)((diffMatrix.array() * diffMatrix.array()).sum() / 2.0f);

    topBlobs[0]->data[0] = loss;

    new (&topBlobs[0]->dataMatrix) Map<MatrixXf>(new float { loss }, 1, 1);
}

void EuclideanLossLayer::backward()
{
    bottomBlobs[0]->diffMatrix = diffMatrix *  1.0f;
    bottomBlobs[1]->diffMatrix = diffMatrix * -1.0f;
}