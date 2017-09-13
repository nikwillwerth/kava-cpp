#ifndef KAVA_BLOB_H
#define KAVA_BLOB_H

#include "Eigen/Dense"

using namespace Eigen;

class Blob
{
public:
    Blob(std::string name);
    Blob(std::string name, int num, int channels, int height, int width);
    Blob(int num, int channels, int height, int width);
    void reshape(int num, int channels, int height, int width);

    std::string name;
    float *data;
    float *diff;
    int count;

    Map<MatrixXf> *dataMatrix;
    Map<MatrixXf> *diffMatrix;
};

#endif //KAVA_BLOB_H