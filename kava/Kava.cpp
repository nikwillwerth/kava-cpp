#include <iostream>
#include "Kava.h"

Kava::Kava()
{
    layers = std::vector<Layer *>();
    nameToLayerMap = std::map<std::string, Layer *>();
    nameToBlobMap  = std::map<std::string, Blob *>();
}

Kava* Kava::addLayer(Layer *layer)
{
    layers.push_back(layer);
    nameToLayerMap[layer->name] = layer;

    for(Blob *topBlob : layer->topBlobs)
    {
        if(nameToBlobMap.count(topBlob->name))
        {
            if(!layer->isInPlace)
            {
                std::cout << "Duplicate blob found in layer " << layer->name << ": " << topBlob->name << std::endl;

                exit(-1);
            }
        }

        nameToBlobMap[topBlob->name] = topBlob;
    }

    return this;
}

void Kava::setUp()
{
    for(int layerIndex = 0; layerIndex < layers.size(); layerIndex++)
    {
        Layer *layer = layers[layerIndex];

        if(layerIndex > 0)
        {
            layer->bottomBlobs = std::vector<Blob *>();

            for(int i = 0; i < layer->bottomBlobNames.size(); i++)
            {
                if(layer->name == layer->bottomBlobNames[i])
                {
                    std::cout << "Bottom blob cannot have the same name as the layer it is in: " << layer->name << std::endl;

                    exit(-1);
                }

                if(!nameToBlobMap.count(layer->bottomBlobNames[i]))
                {
                    std::cout << "Bottom blob " << layer->bottomBlobNames[i] << " not found for layer " << layer->name << std::endl;

                    exit(-1);
                }

                layer->bottomBlobs.push_back(nameToBlobMap[layer->bottomBlobNames[i]]);

                if(layer->isInPlace)
                {
                    layer->topBlobs.push_back(nameToBlobMap[layer->bottomBlobNames[i]]);
                }
            }
        }

        layer->setUp();
    }

    float learningRate = 0.001f;

    int numIterations = 1;

    const clock_t begin_time = clock();

    for(int i = 0; i < numIterations; i++)
    {
        for(long j = 0; j < layers.size(); j++)
        {
            layers[j]->forward();
        }

        for(long j = (layers.size() - 1); j > 0; j--)
        {
            layers[j]->backward();

            if(j == (layers.size() - 1))
            {
                float loss = layers[j]->topBlobs[0]->dataMatrix.data()[0];

                if((i % 100) == 0)
                {
                    //std::cout << layers[layers.size() - 2]->topBlobs[0]->dataMatrix << std::endl;
                    std::cout << "\tloss: " << loss << std::endl << std::endl;
                }

                if(isnan(loss) || isinf(loss))
                {
                    exit(-1);
                }
            }
        }

        for(long j = 0; j < layers.size(); j++)
        {
            if(layers[j]->weightBlobs.size() > 0)
            {
                layers[j]->weightBlobs[0]->updateWeights(learningRate);
            }
        }

        if((i % (numIterations  / 4)) == 0)
        {
            learningRate /= 2;
        }
    }

    float numSeconds = float(clock () - begin_time) / CLOCKS_PER_SEC;

    std::cout << "Time per image for training: " << (numSeconds / numIterations) << std::endl;
    std::cout << "Total time for training:     " << numSeconds << std::endl;
}