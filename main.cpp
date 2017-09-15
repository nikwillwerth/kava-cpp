#include <iostream>
#include "kava/Kava.h"
#include "kava/layers/DataLayer.h"
#include "kava/layers/InnerProductLayer.h"
#include "kava/layers/EuclideanLossLayer.h"

int main()
{
    Kava *kava = new Kava();

    kava->addLayer(new DataLayer("data", 28, 28, 1));
    kava->addLayer(new DataLayer("label", 2048, 1, 1));
    kava->addLayer(new InnerProductLayer("fc1", "data", "fc1", 2048));
    kava->addLayer(new EuclideanLossLayer("loss", "fc1", "label", "loss"));

    kava->setUp();

    return 0;
}