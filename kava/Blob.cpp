#include <iostream>
#include "Blob.h"

Blob::Blob(const std::string &name)
{
    this->name = name;
}

Blob::Blob(const std::string &name, const int channels, const int height, const int width)
{
    this->name = name;

    reshape(channels, height, width);
}

Blob::Blob(const int channels, const int height, const int width)
{
    reshape(channels, height, width);
}

void Blob::reshape(const int channels, const int height, const int width)
{
    this->channels = channels;
    this->height   = height;
    this->width    = width;

    count = channels * height * width;

    data = new float[count];
    diff = new float[count];

    dataMatrix = MatrixXf(count, 1);
    diffMatrix = MatrixXf(count, 1);
}

void Blob::updateWeights(float learningRate)
{
    diffMatrix.resize(dataMatrix.rows(), dataMatrix.cols());

    dataMatrix -= diffMatrix * learningRate;

    //dataMatrix = &dataResult;
    new (&dataMatrix) Map<MatrixXf>(dataMatrix.data(), dataMatrix.rows(), dataMatrix.cols());
}