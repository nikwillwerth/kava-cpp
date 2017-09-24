//
// Created by Nik Willwerth on 9/24/17.
//

#ifndef KAVA_CPP_RANDOMUTILS_H
#define KAVA_CPP_RANDOMUTILS_H

#include <random>

class RandomUtils
{
public:
    static float getRandomGaussian(float mean, float std);
};

#endif //KAVA_CPP_RANDOMUTILS_H