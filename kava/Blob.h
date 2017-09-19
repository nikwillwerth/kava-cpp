#ifndef KAVA_BLOB_H
#define KAVA_BLOB_H

#include "Eigen/Dense"

using namespace Eigen;

class Blob
{
public:
    Blob(const std::string &name);
    Blob(const std::string &name, int channels, int height, int width);
    Blob(int channels, int height, int width);
    void reshape(int channels, int height, int width);
    void updateWeights(float learningRate);

    std::string name;
    float *data;
    float *diff;
    int count;
    int channels;
    int height;
    int width;

    MatrixXf dataMatrix;
    MatrixXf diffMatrix;
};

#endif //KAVA_BLOB_H