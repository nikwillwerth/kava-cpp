#ifndef KAVA_BLOB_H
#define KAVA_BLOB_H

#include "Eigen/Dense"

class blob
{
public:
    Eigen::MatrixXf data;
    Eigen::MatrixXf diff;

    int count;
};

#endif //KAVA_BLOB_H