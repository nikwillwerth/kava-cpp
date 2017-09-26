#include <iostream>
#include "kava/Kava.h"
#include "kava/layers/DataLayer.h"
#include "kava/layers/InnerProductLayer.h"
#include "kava/layers/ReLULayer.h"
#include "kava/layers/SoftmaxWithLossLayer.h"
#include "kava/layers/EuclideanLossLayer.h"
#include "kava/layers/MNISTDataLayer.h"
#include "kava/layers/ConvolutionalLayer.h"

int main()
{
    srand(time(NULL));

    Kava kava = Kava();

    kava.addLayer(new DataLayer("data", 256, 256, 3));
    kava.addLayer(new DataLayer("label", 1, 10, 1));
    //kava.addLayer(new MNISTDataLayer("data", "data", "label", "/Users/nik/CLionProjects/kava-cpp/data/mnist/"));
    //kava.addLayer(new MNISTDataLayer("data", "data", "label", "/home/nik/Desktop/kava-cpp/data/mnist/"));
    //kava.addLayer((new InnerProductLayer("fc1", "data", "fc1", 4096))->setWeightFiller(WeightFiller::Type::MSRA));
    kava.addLayer((new ConvolutionalLayer("fc1", "data", "fc1", 64))->setWeightFiller(WeightFiller::Type::MSRA));
    kava.addLayer(new ReLULayer("relu1", "fc1", "fc1"));
    kava.addLayer((new InnerProductLayer("fc2", "fc1", "fc2", 16))->setWeightFiller(WeightFiller::Type::MSRA));
    kava.addLayer(new ReLULayer("relu2", "fc2", "fc2"));
    kava.addLayer((new InnerProductLayer("fc3", "fc2", "fc3", 10))->setWeightFiller(WeightFiller::Type::Xavier));
    kava.addLayer(new SoftmaxWithLossLayer("loss", "fc3", "label", "loss"));
    //kava.addLayer(new EuclideanLossLayer("loss", "fc3", "label", "loss"));

    kava.setUp();

    int temp;

    std::cin >> temp;

    return 0;
}