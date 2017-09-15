//
// Created by nik on 9/13/17.
//

#include <iostream>
#include "Blob.h"

Blob::Blob(std::string name)
{
    this->name = name;
}

Blob::Blob(std::string name, const int num, const int channels, const int height, const int width)
{
    this->name = name;

    reshape(num, channels, height, width);
}

Blob::Blob(const int num, const int channels, const int height, const int width)
{
    reshape(num, channels, height, width);
}

void Blob::reshape(const int num, const int channels, const int height, const int width)
{
    count = num * channels * height * width;

    data = new float[count];
    diff = new float[count];

    dataMatrix = MatrixXf(count, 1);
    diffMatrix = MatrixXf(count, 1);
}

void Blob::updateWeights(float learningRate)
{
    /*for(int i = 0; i < count; i++)
    {
        std::cout << dataMatrix.data()[i] << ", ";
    }

    std::cout << std::endl;*/

    dataMatrix -= diffMatrix * learningRate;

    //dataMatrix = &dataResult;
    new (&dataMatrix) Map<MatrixXf>(dataMatrix.data(), dataMatrix.rows(), dataMatrix.cols());
}