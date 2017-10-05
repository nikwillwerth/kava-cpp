#include <iostream>
#include "Kava.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include "stb_image_write.h"
#include "stb_image_resize.h"
#include "layers/ConvolutionalLayer.h"

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

void saveLayer(std::vector<Layer *> layers, int layerIndex)
{
    ConvolutionalLayer *convLayer = (ConvolutionalLayer*)layers[layerIndex];

    int k  = convLayer->kernelSize;
    int kk = k * k;

    if(kk == 1)
    {
        const int oneXoneResizeWidth = 28;
        const int oneXoneResizeSize  = oneXoneResizeWidth * oneXoneResizeWidth;

        auto *imageData = new unsigned char[oneXoneResizeSize];

        float min =  1e10;
        float max = -1e10;

        for(int i = 0; i < convLayer->numOutputs; i++)
        {
            float value = convLayer->weightBlobs[0]->dataMatrix.data()[i];

            min = std::min(value, min);
            max = std::max(value, max);
        }

        for(int i = 0; i < convLayer->numOutputs; i++)
        {
            float value = convLayer->weightBlobs[0]->dataMatrix.data()[i];

            auto scaledValue = (unsigned char)(((value - min) / (max - min)) * 255);

            for(int j = 0; j < oneXoneResizeSize; j++)
            {
                std::fill(imageData, (imageData + oneXoneResizeSize), scaledValue);
            }

            std::string filename = "weights/" + convLayer->name + "-" + std::to_string(i) + ".png";

            stbi_write_png(filename.c_str(), oneXoneResizeWidth, oneXoneResizeWidth, 1, imageData, oneXoneResizeWidth);
        }
    }
    else
    {
        for(int i = 0; i < convLayer->numOutputs; i++)
        {
            MatrixXf weight = convLayer->weightBlobs[0]->dataMatrix.col(i).matrix();

            for(int j = 0; j < convLayer->bottomBlobs[0]->channels; j++)
            {
                MatrixXf block = weight.block(j * kk, 0, kk, 1);

                auto *imageData = new unsigned char[kk];

                float min =  1e10;
                float max = -1e10;

                for(int i = 0; i < kk; i++)
                {
                    int index = ((k * (i % k)) + (i /k));

                    float value = block.data()[index];

                    min = std::min(value, min);
                    max = std::max(value, max);
                }

                for(int i = 0; i < kk; i++)
                {
                    int index = ((k * (i % k)) + (i /k));

                    float value = block.data()[index];

                    imageData[i] = (unsigned char)(((value - min) / (max - min)) * 255);
                }

                std::string filename = "weights/" + convLayer->name + "-" + std::to_string(i) + "-" + std::to_string(j) + ".png";


                auto *resizedImageData = new unsigned char[kk * 100];

                int newSize = (k * 10);

                stbir_resize_uint8_generic(imageData, k, k, k, resizedImageData, newSize, newSize, newSize, 1,
                                           0,
                                           STBIR_TYPE_UINT8,
                                           STBIR_EDGE_ZERO,
                                           STBIR_FILTER_BOX,
                                           STBIR_COLORSPACE_SRGB,
                                           nullptr);

                stbi_write_png(filename.c_str(), newSize, newSize, 1, resizedImageData, newSize);
            }
        }
    }
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

    float learningRate = 0.0000f;

    int numIterations = 100;

    const clock_t startTime = clock();

    std::vector<float> forwardTimes, backwardTimes;
    forwardTimes.resize(layers.size(), 0);
    backwardTimes.resize(layers.size(), 0);

    for(int i = 1; i <= numIterations; i++)
    {
        for(long j = 0; j < layers.size(); j++)
        {
            const clock_t thisStartTime = clock();

            layers[j]->forward();

            float numSeconds = float(clock() - thisStartTime) / CLOCKS_PER_SEC;

            forwardTimes[j] += numSeconds;
        }

        for(long j = (layers.size() - 1); j > 0; j--)
        {
            const clock_t thisStartTime = clock();

            layers[j]->backward();

            float numSeconds = float(clock() - thisStartTime) / CLOCKS_PER_SEC;

            backwardTimes[j] += numSeconds;

            if(j == (layers.size() - 1))
            {
                float loss = layers[j]->topBlobs[0]->dataMatrix.data()[0];

                if((i % 1) == 0)
                {

                    std::cout << std::to_string(i) << "/" << std::to_string(numIterations) << std::endl;
                    //std::cout << layers[layers.size() - 3]->topBlobs[0]->dataMatrix << std::endl;
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
            const clock_t thisStartTime = clock();

            if(layers[j]->weightBlobs.size() > 0)
            {
                layers[j]->weightBlobs[0]->updateWeights(learningRate);
            }

            float numSeconds = float(clock() - thisStartTime) / CLOCKS_PER_SEC;

            backwardTimes[j] += numSeconds;
        }

       /* if((i % (numIterations  / 4)) == 0)
        {
            learningRate /= 2;
        }*/
    }

    saveLayer(layers, 1);
    saveLayer(layers, 3);

    float numSeconds = float(clock() - startTime) / CLOCKS_PER_SEC;

    std::cout << "Time per image for training: " << (numSeconds / numIterations) << std::endl;
    std::cout << "Total time for training:     " << numSeconds << std::endl;



    float forwardSum = 0;

    for(int i = 0; i < layers.size(); i++)
    {
        forwardSum += (forwardTimes[i] / numIterations);
    }

    std::cout << std::endl << "Forward: " << forwardSum << std::endl;

    for(int i = 0; i < layers.size(); i++)
    {
        std::cout << "\t" << layers[i]->name << " - " << (forwardTimes[i] / numIterations) << std::endl;
    }

    float backwardSum = 0;

    for(int i = 0; i < layers.size(); i++)
    {
        backwardSum += (backwardTimes[i] / numIterations);
    }

    std::cout << std::endl << "Backward: " << backwardSum << std::endl;

    for(int i = 0; i < layers.size(); i++)
    {
        std::cout << "\t" << layers[i]->name << " - " << (backwardTimes[i] / numIterations) << std::endl;
    }
}