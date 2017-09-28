#include <iostream>
#include "SoftmaxWithLossLayer.h"

#define EPS 1e-6
#define ONE_MINUS_EPS 1-1e-6

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
    MatrixXf input = bottomBlobs[0]->dataMatrix;

    //softmax
    float maxInputValue = input.maxCoeff();

    MatrixXf maxInputValueMatrix = MatrixXf::Constant(input.rows(), input.cols(), maxInputValue);

    input -= maxInputValueMatrix;

    //regularization
    MatrixXf inputExponential = input.array().exp();

    float inputExponentialSum = inputExponential.sum();

    MatrixXf softmax = inputExponential / inputExponentialSum;
    softmax = softmax.cwiseMin(ONE_MINUS_EPS);
    softmax = softmax.cwiseMax(EPS);

    //cross entropy loss
    softmax.resize(bottomBlobs[1]->dataMatrix.rows(), bottomBlobs[1]->dataMatrix.cols());

    MatrixXf epsMatrix = MatrixXf::Constant(bottomBlobs[1]->dataMatrix.rows(), bottomBlobs[1]->dataMatrix.cols(), EPS);

    diffMatrix = softmax - epsMatrix - bottomBlobs[1]->dataMatrix;

    MatrixXf logLossMatrix = softmax;

    for(int i = 0; i < logLossMatrix.size(); i++)
    {
        if(logLossMatrix.data()[i] != 0)
        {
            logLossMatrix.data()[i] = logf(logLossMatrix.data()[i]);
        }
        else if(logLossMatrix.data()[i] == EPS)
        {
            logLossMatrix.data()[i] = 0;
        }
    }

    logLossMatrix.resize(bottomBlobs[1]->dataMatrix.rows(), bottomBlobs[1]->dataMatrix.cols());

    //y log(y')
    MatrixXf yLogYPrime = bottomBlobs[1]->dataMatrix.cwiseProduct(logLossMatrix);

    MatrixXf onesMatrix = MatrixXf::Ones(bottomBlobs[1]->dataMatrix.rows(), bottomBlobs[1]->dataMatrix.cols());

    //1 - y
    MatrixXf oneMinusLabel = onesMatrix - bottomBlobs[1]->dataMatrix;

    //1 - y'
    MatrixXf oneMinusPrediction = MatrixXf::Ones(softmax.rows(), softmax.cols()) - softmax;

    //(1-y)log(1-y')
    MatrixXf oneMinusLogOneMinus = oneMinusLabel.cwiseProduct(oneMinusPrediction.array().log().matrix());

    //loss
    topBlobs[0]->data[0] = (yLogYPrime + oneMinusLogOneMinus).sum() / -yLogYPrime.size();
    topBlobs[0]->putDataIntoMatrix();
}

void SoftmaxWithLossLayer::backward()
{
    bottomBlobs[0]->diffMatrix =  diffMatrix;
    bottomBlobs[1]->diffMatrix = -diffMatrix;
}