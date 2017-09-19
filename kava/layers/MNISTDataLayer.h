#ifndef KAVA_CPP_MNISTDATALAYER_H
#define KAVA_CPP_MNISTDATALAYER_H

#include "Layer.h"

class MNISTDataLayer : public Layer
{
public:
    MNISTDataLayer(const std::string name, const std::string dataName, const std::string labelName, const std::string mnistDataDirectory);

    void setUp();
    void forward();

private:
    std::vector<std::vector<MatrixXf>> trainingData;
    std::vector<std::vector<MatrixXf>> testingData;
    std::string mnistDataDirectory;

    int currentTrainingIndex = 0;
};

#endif //KAVA_CPP_MNISTDATALAYER_H