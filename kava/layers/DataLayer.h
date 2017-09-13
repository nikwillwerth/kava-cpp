#ifndef KAVA_DATALAYER_H
#define KAVA_DATALAYER_H

#include "Layer.h"

class DataLayer : public Layer
{
public:
    DataLayer(std::string name, int width, int height, int channels);

    void setUp();
    void forward();

    int width;
    int height;
    int channels;
};

#endif //KAVA_DATALAYER_H