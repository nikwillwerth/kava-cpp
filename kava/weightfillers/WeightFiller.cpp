#include "WeightFiller.h"
#include "ConstantWeightFiller.h"
#include "GaussianWeightFiller.h"
#include "MSRAWeightFiller.h"
#include "XavierWeightFiller.h"

WeightFiller* WeightFiller::getWeightFillerWithType(WeightFiller::Type type)
{
    WeightFiller *weightFiller;

    switch(type)
    {
        case WeightFiller::Type::Constant: weightFiller = new ConstantWeightFiller(); break;
        case WeightFiller::Type::Gaussian: weightFiller = new GaussianWeightFiller(); break;
        case WeightFiller::Type::MSRA:     weightFiller = new MSRAWeightFiller();     break;
        case WeightFiller::Type::Xavier:   weightFiller = new XavierWeightFiller();   break;
        default: break;
    }

    return weightFiller;
}