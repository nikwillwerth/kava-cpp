#include <iostream>
#include "kava/Kava.h"
#include "kava/layers/DataLayer.h"
#include "kava/layers/InnerProductLayer.h"
#include "kava/layers/ReLULayer.h"
#include "kava/layers/SoftmaxWithLossLayer.h"
#include "kava/layers/EuclideanLossLayer.h"

int main()
{
    Kava *kava = new Kava();

    kava->addLayer(new DataLayer("data", 64, 64, 1));
    kava->addLayer(new DataLayer("label", 10, 1, 1));
    kava->addLayer(new InnerProductLayer("fc1", "data", "fc1", 256));
    kava->addLayer(new ReLULayer("relu1", "fc1", "relu1"));
    kava->addLayer(new InnerProductLayer("fc2", "relu1", "fc2", 10));
    kava->addLayer(new SoftmaxWithLossLayer("loss", "fc2", "label", "loss"));

    kava->setUp();

    return 0;
}