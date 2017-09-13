#include "Kava.h"

Kava::Kava()
{
    layers = std::vector<Layer *>();
}

Kava* Kava::addLayer(Layer *layer)
{
    layers.push_back(layer);

    return this;
}

void Kava::setUp()
{
    for(int layerIndex = 0; layerIndex < layers.size(); layerIndex++)
    {
        Layer *layer = layers[layerIndex];

        if(layerIndex > 0)
        {

        }

        layer->setUp();
    }
}