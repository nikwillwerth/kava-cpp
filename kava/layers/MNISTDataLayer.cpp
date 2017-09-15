//
// Created by Nik Willwerth on 9/15/17.
//

#include <iostream>
#include "MNISTDataLayer.h"

const std::string MNIST_URL         = "http://yann.lecun.com/exdb/mnist/";
const std::string TRAIN_IMAGES_NAME = "train-images-idx3-ubyte.gz";
const std::string TRAIN_LABELS_NAME = "train-labels-idx1-ubyte.gz";
const std::string TEST_IMAGES_NAME  = "t10k-images-idx3-ubyte.gz";
const std::string TEST_LABELS_NAME  = "t10k-labels-idx1-ubyte.gz";

void getMnistDataset();

MNISTDataLayer::MNISTDataLayer(const std::string name, const std::string dataName, const std::string labelName)
{
    this->name = name;

    topBlobs = std::vector<Blob *>();

    topBlobs.push_back(new Blob(name, 1, 28, 28));
    topBlobs.push_back(new Blob(name, 1, 1, 10));
}

void MNISTDataLayer::setUp()
{
    getMnistDataset();
}

void MNISTDataLayer::forward()
{

}

void getMnistDataset()
{
    std::cout << "Downloading MNIST dataset..." << std::endl;
}