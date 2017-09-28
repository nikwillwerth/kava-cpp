#include <iostream>
#include "DataLayer.h"
#include "../utils/MathUtils.h"

DataLayer::DataLayer(std::string name, int width, int height, int channels)
{
    this->name     = name;
    this->width    = width;
    this->height   = height;
    this->channels = channels;

    topBlobs = std::vector<Blob *>();

    topBlobs.push_back(new Blob(name, channels, height, width));
}

void DataLayer::setUp()
{
    std::cout << "Setting up data layer..." << std::endl;

    if(name != "label")
    {
        for(int i = 0; i < topBlobs[0]->count; i++)
        {
            topBlobs[0]->data[i] = i;
        }
    }
    else
    {
        for(int i = 0; i < topBlobs[0]->count; i++)
        {
            topBlobs[0]->data[i] = 1;
        }

        topBlobs[0]->data[3] = 1;
    }

    topBlobs[0]->putDataIntoMatrix();
}

void DataLayer::forward()
{

}