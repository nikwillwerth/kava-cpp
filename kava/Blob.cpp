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

    dataMatrix = new Map<MatrixXf>(data, count, 1);
    diffMatrix = new Map<MatrixXf>(diff, count, 1);
}