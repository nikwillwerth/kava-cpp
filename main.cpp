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

    //kava.addLayer(new MNISTDataLayer("data", "data", "label", "/Users/nik/CLionProjects/kava-cpp/data/mnist/"));
    kava.addLayer(new MNISTDataLayer("data", "data", "label", "/home/nik/Desktop/kava-cpp/data/mnist/"));
    kava.addLayer(((new ConvolutionalLayer("conv1", "data", "conv1", 16))->setKernelSize(11)->setStride(1))->setWeightFiller(WeightFiller::Type::MSRA));
    kava.addLayer(new ReLULayer("relu1", "conv1", "conv1"));
    kava.addLayer(new MaxPoolingLayer("pool1", "conv1", "pool1"));
    kava.addLayer(((new ConvolutionalLayer("conv2", "pool1", "conv2", 16))->setKernelSize(3))->setWeightFiller(WeightFiller::Type::MSRA));
    kava.addLayer(new ReLULayer("relu2", "conv2", "conv2"));
    /*kava.addLayer(((new ConvolutionalLayer("conv3", "conv2", "conv3", 16))->setKernelSize(5))->setWeightFiller(WeightFiller::Type::MSRA));
    kava.addLayer(new ReLULayer("relu3", "conv3", "conv3"));
    kava.addLayer(((new ConvolutionalLayer("conv4", "conv3", "conv4", 16))->setKernelSize(3))->setWeightFiller(WeightFiller::Type::MSRA));
    kava.addLayer(new ReLULayer("relu4", "conv4", "conv4"));
    kava.addLayer((new InnerProductLayer("fc1", "conv1", "fc1", 512))->setWeightFiller(WeightFiller::Type::Xavier));*/
    kava.addLayer((new InnerProductLayer("fc2", "pool1", "fc2", 10))->setWeightFiller(WeightFiller::Type::Xavier));
    kava.addLayer(new SoftmaxWithLossLayer("loss", "fc2", "label", "loss"));

    kava.setUp();

    //int temp;

    //std::cin >> temp;

    return 0;
}