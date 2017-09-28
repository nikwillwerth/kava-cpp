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

    //std::cout << diffMatrix << std::endl;

    dataMatrix -= diffMatrix * learningRate;

    putDataIntoMatrix();
}

void Blob::putDataIntoMatrix()
{
    new (&dataMatrix) Map<MatrixXf>(data, height, width * channels);
}

void Blob::putDiffIntoMatrix()
{
    new (&diffMatrix) Map<MatrixXf>(diff, height, width * channels);
}