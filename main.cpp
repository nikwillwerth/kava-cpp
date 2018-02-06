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
#include "kava/layers/AveragePoolingLayer.h"

int main()
{
    srand(time(NULL));

    Kava kava = Kava();

    kava.addLayer(new MNISTDataLayer("data", "data", "label", "/Users/nik/CLionProjects/kava-cpp/data/mnist/"));

    kava.addLayer(((new ConvolutionalLayer("conv1", "data", "conv1", 32))->setKernelSize(5)->setStride(1)->setPadding(0))->setWeightFiller(WeightFiller::Type::MSRA));
    kava.addLayer(new ReLULayer("relu1", "conv1", "conv1"));
    kava.addLayer(((new MaxPoolingLayer("max1", "conv1", "max1"))->setKernelSize(2)));

    kava.addLayer(((new ConvolutionalLayer("conv2", "max1", "conv2", 32))->setKernelSize(3)->setStride(1)->setPadding(0))->setWeightFiller(WeightFiller::Type::MSRA));
    kava.addLayer(new ReLULayer("relu2", "conv2", "conv2"));
    kava.addLayer(((new MaxPoolingLayer("max2", "conv2", "max2"))->setKernelSize(2)));

    kava.addLayer(((new ConvolutionalLayer("conv3", "max2", "conv3", 32))->setKernelSize(3)->setStride(1)->setPadding(0))->setWeightFiller(WeightFiller::Type::MSRA));
    kava.addLayer(new ReLULayer("relu3", "conv3", "conv3"));

    kava.addLayer(((new AveragePoolingLayer("ip1", "conv3", "ip1"))->setKernelSize(3)));
    kava.addLayer((new InnerProductLayer("ip2", "ip1", "ip2", 10))->setWeightFiller(WeightFiller::Type::MSRA));
    kava.addLayer(new SoftmaxWithLossLayer("loss", "ip2", "label", "loss"));

    kava.setLearningRate((float)0.000000001);
    kava.setUp();
    kava.train(nullptr);

    //int temp;

    //std::cin >> temp;

    return 0;
}