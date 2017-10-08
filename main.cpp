#include <iostream>
#include "kava/Kava.h"
#include "kava/layers/DataLayer.h"
#include "kava/layers/InnerProductLayer.h"
#include "kava/layers/MaxPoolingLayer.h"
#include "kava/layers/ReLULayer.h"
#include "kava/layers/SoftmaxWithLossLayer.h"
#include "kava/layers/EuclideanLossLayer.h"
#include "kava/layers/MNISTDataLayer.h"
#include "kava/layers/ConvolutionalLayer.h"

int main()
{
    srand(time(NULL));

    Kava kava = Kava();

    kava.addLayer(new MNISTDataLayer("data", "data", "label", "/Users/nik/CLionProjects/kava-cpp/data/mnist/"));
    //kava.addLayer(new MNISTDataLayer("data", "data", "label", "/home/nik/Desktop/kava-cpp/data/mnist/"));
    kava.addLayer(((new ConvolutionalLayer("conv1", "data", "conv1", 20))->setKernelSize(5)->setStride(1))->setWeightFiller(WeightFiller::Type::Xavier));
    //kava.addLayer(new ReLULayer("relu1", "conv1", "conv1"));
    kava.addLayer(new MaxPoolingLayer("pool1", "conv1", "pool1"));
    kava.addLayer(((new ConvolutionalLayer("conv2", "pool1", "conv2", 50))->setKernelSize(5)->setStride(1))->setWeightFiller(WeightFiller::Type::Xavier));
    //kava.addLayer(new ReLULayer("relu2", "conv2", "conv2"));
    kava.addLayer(new MaxPoolingLayer("pool2", "conv2", "pool2"));
    kava.addLayer((new InnerProductLayer("ip1", "pool2", "ip1", 500))->setWeightFiller(WeightFiller::Type::Xavier));
    kava.addLayer(new ReLULayer("relu3", "ip1", "ip1"));
    kava.addLayer((new InnerProductLayer("ip2", "ip1", "ip2", 10))->setWeightFiller(WeightFiller::Type::Xavier));
    kava.addLayer(new SoftmaxWithLossLayer("loss", "ip2", "label", "loss"));

    kava.setUp();
    kava.train(nullptr);

    //int temp;

    //std::cin >> temp;

    return 0;
}