//
#include <iostream>
#include <fstream>
#include <iterator>
#include "MNISTDataLayer.h"
#include "../utils/FileUtils.h"

const std::string TRAIN_IMAGES_NAME = "train-images-idx3-ubyte";
const std::string TRAIN_LABELS_NAME = "train-labels-idx1-ubyte";
const std::string TEST_IMAGES_NAME  = "t10k-images-idx3-ubyte";
const std::string TEST_LABELS_NAME  = "t10k-labels-idx1-ubyte";

int charsToInt(std::vector<unsigned char> buffer, int offset)
{
    return (buffer[offset] << 24) | (buffer[offset + 1] << 16) | (buffer[offset + 2] << 8) | (buffer[offset + 3]);
}

void readImageFile(const std::string filename, std::vector<std::vector<MatrixXf>> &dataHolder)
{
    std::ifstream file(filename, std::ios::binary);

    file.unsetf(std::ios::skipws);

    std::istream_iterator<unsigned char> begin(file), end;

    std::vector<unsigned char> buffer(begin, end);

    file.close();

    int magicNumber    = charsToInt(buffer, 0);
    int numberOfImages = charsToInt(buffer, 4);
    int numberOfRows   = charsToInt(buffer, 8);
    int numberOfCols   = charsToInt(buffer, 12);

    int imageSize = numberOfRows * numberOfCols;

    for(int offset = 16; offset < buffer.size(); offset += imageSize)
    {
        MatrixXf thisMatrix = MatrixXf(numberOfRows, numberOfCols);

        for(int index = 0; index < imageSize; index++)
        {
            int thisIndex = (index % numberOfRows) * numberOfRows + (index / numberOfRows);

            thisMatrix.data()[index] = static_cast<float>(buffer[offset + thisIndex]) / 255.0f;
        }

        std::vector<MatrixXf> thisData = std::vector<MatrixXf>();
        thisData.push_back(thisMatrix);

        dataHolder.push_back(thisData);
    }
}

void readLabelFile(const std::string filename, std::vector<std::vector<MatrixXf>> &dataHolder)
{
    std::ifstream file(filename, std::ios::binary);

    file.unsetf(std::ios::skipws);

    std::istream_iterator<unsigned char> begin(file), end;

    std::vector<unsigned char> buffer(begin, end);

    file.close();

    int magicNumber   = charsToInt(buffer, 0);
    int numberOfItems = charsToInt(buffer, 4);

    for(int offset = 8; offset < buffer.size(); offset++)
    {
        int classIndex = static_cast<int>(buffer[offset]);

        MatrixXf thisMatrix = MatrixXf::Zero(1, 10);
        thisMatrix.data()[classIndex] = 1;

        dataHolder[offset - 8].push_back(thisMatrix);
    }
}

MNISTDataLayer::MNISTDataLayer(const std::string name, const std::string dataName, const std::string labelName, const std::string mnistDataDirectory)
{
    this->name               = name;
    this->mnistDataDirectory = mnistDataDirectory;

    trainingData = std::vector<std::vector<MatrixXf>>();
    testingData  = std::vector<std::vector<MatrixXf>>();
    topBlobs     = std::vector<Blob *>();

    topBlobs.push_back(new Blob(dataName, 1, 28, 28));
    topBlobs.push_back(new Blob(labelName, 1, 1, 10));
}

void MNISTDataLayer::setUp()
{
    std::cout << "Setting up MNIST data layer..." << std::endl;

    bool trainingImagesExist = FileUtils::doesFileExist(mnistDataDirectory + TRAIN_IMAGES_NAME);
    bool trainingLabelsExist = FileUtils::doesFileExist(mnistDataDirectory + TRAIN_LABELS_NAME);
    bool testingImagesExist  = FileUtils::doesFileExist(mnistDataDirectory + TEST_IMAGES_NAME);
    bool testingLabelsExist  = FileUtils::doesFileExist(mnistDataDirectory + TEST_LABELS_NAME);

    if(!trainingImagesExist || !trainingLabelsExist || !testingImagesExist || !testingLabelsExist)
    {
        std::cout << "Error! MNIST dataset not found in given path! Please run get_mnist script in the data directory to download the dataset." << std::endl;

        exit(-1);
    }

    readImageFile(mnistDataDirectory + TRAIN_IMAGES_NAME, trainingData);
    readImageFile(mnistDataDirectory + TEST_IMAGES_NAME,  testingData);
    readLabelFile(mnistDataDirectory + TRAIN_LABELS_NAME, trainingData);
    readLabelFile(mnistDataDirectory + TEST_LABELS_NAME,  testingData);
}

void MNISTDataLayer::forward()
{
    topBlobs[0]->dataMatrix = trainingData[currentTrainingIndex][0];
    topBlobs[1]->dataMatrix = trainingData[currentTrainingIndex][1];

    currentTrainingIndex++;

    if(currentTrainingIndex == trainingData.size())
    {
        currentTrainingIndex = 0;
    }
}