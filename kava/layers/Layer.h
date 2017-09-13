#ifndef KAVA_LAYER_H
#define KAVA_LAYER_H

class Layer
{
public:
    virtual void setUp()    {}
    virtual void forward()  {}
    virtual void backward() {}
};

#endif //KAVA_LAYER_H