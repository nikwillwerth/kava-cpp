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
    void setLearningRate(float learningRate);

private:
    std::vector<Layer *> layers;
    std::map<std::string, Layer *> nameToLayerMap;
    std::map<std::string, Blob *> nameToBlobMap;
    
    float learningRate = 0.01f;
};

#endif //KAVA_KAVA_H
