#ifndef KAVA_KAVA_H
#define KAVA_KAVA_H

#include <vector>
#include <map>
#include "layers/Layer.h"

class Kava
{
public:
    Kava();
    Kava* addLayer(Layer *layer);
    void setUp();
    void train(std::function<void (const float)> lossCallback);

private:
    std::vector<Layer *> layers;
    std::map<std::string, Layer *> nameToLayerMap;
    std::map<std::string, Blob *> nameToBlobMap;
    
    float learningRate = 0.001f;
};

#endif //KAVA_KAVA_H
