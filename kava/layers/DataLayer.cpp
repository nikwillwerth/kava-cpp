#include <iostream>
#include "DataLayer.h"

DataLayer::DataLayer(std::string name, int width, int height, int channels)
{
    this->name     = name;
    this->width    = width;
    this->height   = height;
    this->channels = channels;

    bottomBlobs = std::vector<Blob *>();
    topBlobs    = std::vector<Blob *>();
    weightBlobs = std::vector<Blob *>();

    topBlobs.push_back(new Blob(name, 1, channels, height, width));
}

void DataLayer::setUp()
{
    std::cout << "Setting up data layer..." << std::endl;
}

void DataLayer::forward()
{
    float *data = new float[topBlobs[0]->count];

    for(int i = 0; i < topBlobs[0]->count; i++)
    {
        data[i] = i;
    }

    topBlobs[0]->dataMatrix = new Map<MatrixXf>(data, width, height);
}