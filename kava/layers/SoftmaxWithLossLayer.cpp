#include <iostream>
#include "SoftmaxWithLossLayer.h"

SoftmaxWithLossLayer::SoftmaxWithLossLayer(std::string name, std::string bottomBlobOneName, std::string bottomBlobTwoName, std::string topBlobName)
{
    this->name = name;

    bottomBlobNames = std::vector<std::string>();
    topBlobs        = std::vector<Blob *>();

    bottomBlobNames.push_back(bottomBlobOneName);
    bottomBlobNames.push_back(bottomBlobTwoName);
    topBlobs.push_back(new Blob(topBlobName, 1, 1, 1));
}

void SoftmaxWithLossLayer::setUp()
{
    std::cout << "Setting up softmax with loss layer..." << std::endl;

    if(bottomBlobs[0]->count != bottomBlobs[1]->count)
    {
        std::cout << "Blob sizes do not match in layer " << name << "(" << bottomBlobs[0]->count << " vs. " << bottomBlobs[1]->count << ")" << std::endl;
    }
}

float epsilon = 1.1755e-38;

void SoftmaxWithLossLayer::forward()
{
    float max      = bottomBlobs[0]->dataMatrix.array().maxCoeff();
    int   maxIndex = -1;

    MatrixXf shift = MatrixXf::Ones(bottomBlobs[0]->dataMatrix.rows(), bottomBlobs[0]->dataMatrix.cols());
    shift *= max;

    for(int i = 0; i < bottomBlobs[0]->count; i++)
    {
        if(bottomBlobs[0]->dataMatrix.data()[i] == max)
        {
            maxIndex = i;
        }
    }

    MatrixXf input = bottomBlobs[0]->dataMatrix - shift;

    MatrixXf exp = input.array().exp();

    float sum_j = exp.array().sum();

    MatrixXf p = exp / sum_j;

    MatrixXf eps = MatrixXf::Ones(p.rows(), p.cols());
    eps *= epsilon;

    p += eps;

    p.resize(bottomBlobs[1]->dataMatrix.rows(), bottomBlobs[1]->dataMatrix.cols());

    diffMatrix = p - bottomBlobs[1]->dataMatrix;



    MatrixXf lossArray = -p.array().log();

    float loss = fabs(lossArray.data()[maxIndex]);

    new (&topBlobs[0]->dataMatrix) Map<MatrixXf>(new float { loss }, 1, 1);
}

void SoftmaxWithLossLayer::backward()
{
    bottomBlobs[0]->diffMatrix = diffMatrix;
    bottomBlobs[1]->diffMatrix = diffMatrix;
}