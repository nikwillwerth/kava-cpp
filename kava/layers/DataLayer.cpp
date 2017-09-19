#include <iostream>
#include "DataLayer.h"

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
            float randy = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

            topBlobs[0]->data[i] = randy;
        }
    }
    else
    {
        for(int i = 0; i < topBlobs[0]->count; i++)
        {
            topBlobs[0]->data[i] = 0;
        }

        int index = 9;//(int)(static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 9.9f);

        topBlobs[0]->data[index] = 1;
    }

    new (&topBlobs[0]->dataMatrix) Map<MatrixXf>(topBlobs[0]->data, width, height);
}

void DataLayer::forward()
{

}