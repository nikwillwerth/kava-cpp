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
    /*std::cout << "\t\t\t\t" << bottomBlobs[0] << std::endl;

    Map<RowVectorXf> a(bottomBlobs[0]->dataMatrix->data(), bottomBlobs[0]->dataMatrix->size());

    std::cout << bottomBlobs[0]->name << std::endl;

    for(int i = 0; i < a.size(); i++)
    {
        std::cout << a[i] << ", ";
    }

    std::cout << std::endl;*/

    Map<RowVectorXf> bottomOne(bottomBlobs[0]->dataMatrix.data(), bottomBlobs[0]->count);
    Map<RowVectorXf> bottomTwo(bottomBlobs[1]->dataMatrix.data(), bottomBlobs[1]->count);

    diffMatrix = bottomOne - bottomTwo;

    float loss = (float)((diffMatrix.array() * diffMatrix.array()).sum() / 2.0f);;

    topBlobs[0]->data[0] = loss;

    new (&topBlobs[0]->dataMatrix) Map<MatrixXf>(new float { loss }, 1, 1);
}

void EuclideanLossLayer::backward()
{
    //new (bottomBlobs[0]->diffMatrix) Map<MatrixXf>(diffMatrix.data(), diffMatrix.rows(), diffMatrix.cols());

    bottomBlobs[0]->diffMatrix = diffMatrix *  1.0f;//Map<MatrixXf>(diffMatrix.data(), diffMatrix.rows(), diffMatrix.cols());
    bottomBlobs[1]->diffMatrix = diffMatrix * -1.0f;//Map<MatrixXf>(diffMatrix.data(), diffMatrix.rows(), diffMatrix.cols());

    /*std::cout << "backward" << std::endl;

    Map<RowVectorXf> a(diffMatrix.data(), diffMatrix.size());
    Map<RowVectorXf> b(bottomBlobs[0]->diffMatrix->data(), bottomBlobs[0]->diffMatrix->size());
    Map<RowVectorXf> c(bottomBlobs[1]->diffMatrix->data(), bottomBlobs[1]->diffMatrix->size());

    std::cout << "1." << std::endl << "\t";

    for(int i = 0; i < a.size(); i++)
    {
        std::cout << a[i] << ", ";
    }

    std::cout << std::endl;

    std::cout << "2." << std::endl << "\t";

    for(int i = 0; i < b.size(); i++)
    {
        std::cout << b[i] << ", ";
    }

    std::cout << std::endl;

    std::cout << "3." << std::endl << "\t";

    for(int i = 0; i < c.size(); i++)
    {
        std::cout << c[i] << ", ";
    }

    std::cout << std::endl;*/

    //std::cout << "\t\t1. " <<                  diffMatrix << std::endl;
    //std::cout << "\t\t2. " << *bottomBlobs[0]->diffMatrix << std::endl;
    //std::cout << "\t\t3. " << *bottomBlobs[1]->diffMatrix << std::endl;
}