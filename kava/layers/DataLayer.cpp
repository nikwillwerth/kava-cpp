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
    for(int i = 0; i < topBlobs[0]->count; i++)
    {
        topBlobs[0]->data[i] = i % 2;
    }

    new (&topBlobs[0]->dataMatrix) Map<MatrixXf>(topBlobs[0]->data, width, height);
}