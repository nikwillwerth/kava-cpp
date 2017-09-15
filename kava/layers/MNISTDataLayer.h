#ifndef KAVA_CPP_MNISTDATALAYER_H
#define KAVA_CPP_MNISTDATALAYER_H

#include "Layer.h"

class MNISTDataLayer : public Layer
{
public:
    MNISTDataLayer(const std::string name, const std::string dataName, const std::string labelName);

    void setUp();
    void forward();

private:
    std::vector<std::vector<MatrixXf>> trainingData;
    std::vector<std::vector<MatrixXf>> testingData;
};

#endif //KAVA_CPP_MNISTDATALAYER_H