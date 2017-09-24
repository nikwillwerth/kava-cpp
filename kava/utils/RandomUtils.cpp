#include <iostream>
#include "RandomUtils.h"

static std::random_device seed;

static std::seed_seq seed_sequence{ seed(), seed(), seed(), seed(), seed(), seed(), seed(), seed() };
static std::mt19937 gen(seed_sequence);
static std::normal_distribution<> normalDistribution(0, 1);

float RandomUtils::getRandomGaussian(float mean, float std)
{
    return (float)((normalDistribution(gen) * std) + mean);
}