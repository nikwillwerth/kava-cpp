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
    topBlobs[0]->reshape(   1, 1, 1,          numOutputs);

    for(int i = 0; i < weightBlobs[0]->count; i++)
    {
        weightBlobs[0]->data[i] = 0.01f;
        weightBlobs[0]->diff[i] = 0;
    }

    new (&weightBlobs[0]->dataMatrix) Map<MatrixXf>(weightBlobs[0]->data, numOutputs, numInputs);
    new (&weightBlobs[0]->diffMatrix) Map<MatrixXf>(weightBlobs[0]->diff, numOutputs, numInputs);
}

void InnerProductLayer::forward()
{
    //Map<RowVectorXf> bottom( bottomBlobs[0]->data, bottomBlobs[0]->count);
    //Map<MatrixXf>    weights(weightBlobs[0]->data, numInputs, numOutputs);

    bottomBlobs[0]->dataMatrix.resize(1,         numInputs);
    //weightBlobs[0]->dataMatrix->resize(numInputs, numOutputs);

    weightBlobs[0]->dataMatrix.transposeInPlace();

    topBlobs[0]->dataMatrix = bottomBlobs[0]->dataMatrix * weightBlobs[0]->dataMatrix;

    //new (topBlobs[0]->dataMatrix) Map<MatrixXf>(result.data(), 1, numOutputs);

    /*std::cout << "\t\t\t\t" << topBlobs[0] << std::endl;

    Map<RowVectorXf> a(topBlobs[0]->dataMatrix->data(), topBlobs[0]->dataMatrix->size());

    std::cout << "1." << std::endl << "\t";

    for(int i = 0; i < a.size(); i++)
    {
        std::cout << a[i] << ", ";
    }

    std::cout << std::endl;*/
}

void InnerProductLayer::backward()
{
    //weightBlobs[0]->dataMatrix->resize(numOutputs, numInputs);
    weightBlobs[0]->dataMatrix.transposeInPlace();

    MatrixXf dInput = topBlobs[0]->diffMatrix * weightBlobs[0]->dataMatrix;

    new (&bottomBlobs[0]->diffMatrix) Map<MatrixXf>(dInput.data(), dInput.rows(), dInput.cols());

    //topBlobs[0]->diffMatrix->transposeInPlace();
    topBlobs[0]->diffMatrix.resize(topBlobs[0]->diffMatrix.cols(), topBlobs[0]->diffMatrix.rows());
    bottomBlobs[0]->dataMatrix.resize(1, numInputs);

    //std::cout << topBlobs[0]->diffMatrix->rows() << "x" << topBlobs[0]->diffMatrix->cols() << std::endl;
    //std::cout << bottomBlobs[0]->dataMatrix->rows() << "x" << bottomBlobs[0]->dataMatrix->cols() << std::endl;

    MatrixXf dWeights = topBlobs[0]->diffMatrix * bottomBlobs[0]->dataMatrix;

    new (&weightBlobs[0]->diffMatrix) Map<MatrixXf>(dWeights.data(), dWeights.rows(), dWeights.cols());
}