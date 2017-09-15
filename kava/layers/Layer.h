#ifndef KAVA_LAYER_H
#define KAVA_LAYER_H

#include "../Blob.h"
#include <string>
#include <vector>

class Layer
{
public:
    virtual void setUp()    {}
    virtual void forward()  {}
    virtual void backward() {}

    std::string         name;
    std::vector<std::string> bottomBlobNames;
    std::vector<Blob *> bottomBlobs;
    std::vector<Blob *> topBlobs;
    std::vector<Blob *> weightBlobs;
};

#endif //KAVA_LAYER_H