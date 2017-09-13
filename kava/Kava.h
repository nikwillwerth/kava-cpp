#ifndef KAVA_KAVA_H
#define KAVA_KAVA_H

#include <vector>
#include "layers/Layer.h"

class Kava
{
    std::vector<Layer *> layers;

    Kava();
    Kava* addLayer(Layer *layer);
    void setUp();
};

#endif //KAVA_KAVA_H