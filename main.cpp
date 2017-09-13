#include <iostream>
#include "kava/Kava.h"
#include "kava/layers/DataLayer.h"
#include "kava/layers/InnerProductLayer.h"

int main()
{
    Kava *kava = new Kava();

    kava->addLayer(new DataLayer("data", 28, 28, 1));
    kava->addLayer(new InnerProductLayer("fc1", "data", "fc1", 16));

    kava->setUp();

    return 0;
}