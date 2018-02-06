#include <iostream>
#include "RandomUtils.h"

static std::random_device seed;

static std::seed_seq seed_sequence{ seed(), seed(), seed(), seed(), seed(), seed(), seed(), seed() };
static std::mt19937 gen(seed_sequence);

float RandomUtils::getRandomGaussian(float mean, float std)
{
    std::normal_distribution<> thisNormalDistribution(mean, std);

    return (float)thisNormalDistribution(gen);
}

float RandomUtils::getRandomUniform(float mean, float std)
{
    std::uniform_real_distribution<> thisUniformDistribution(mean, std);

    return (float)thisUniformDistribution(gen);
}