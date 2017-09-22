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

void SoftmaxWithLossLayer::forward()
{
    MatrixXf exp = bottomBlobs[0]->dataMatrix.array().exp();

    float denominator = exp.sum();

    MatrixXf softmax = exp / denominator;

    softmax.resize(bottomBlobs[1]->dataMatrix.rows(), bottomBlobs[1]->dataMatrix.cols());

    bottomBlobs[0]->diffMatrix.noalias() = softmax - bottomBlobs[1]->dataMatrix;

    MatrixXf log = softmax.array().log();

    log.resize(bottomBlobs[1]->dataMatrix.rows(), bottomBlobs[1]->dataMatrix.cols());

    MatrixXf mul = bottomBlobs[1]->dataMatrix.cwiseProduct(log);

    MatrixXf inverseTargets = MatrixXf::Ones(bottomBlobs[1]->dataMatrix.rows(), bottomBlobs[1]->dataMatrix.cols()) - bottomBlobs[1]->dataMatrix;

    MatrixXf inversePredictions = (MatrixXf::Ones(softmax.rows(), softmax.cols()) * (1.0f + 1.0e-6)) - softmax;

    MatrixXf b = inverseTargets.cwiseProduct(inversePredictions.array().log().matrix());

    float loss = (mul + b).sum() / -mul.size();

    new (&topBlobs[0]->dataMatrix) Map<MatrixXf>(new float { loss }, 1, 1);
}

void SoftmaxWithLossLayer::backward()
{
    //bottomBlobs[0]->diffMatrix = diffMatrix;
    //bottomBlobs[1]->diffMatrix = diffMatrix;
}