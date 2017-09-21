//
// Created by nik on 9/13/17.
//

#include "Layer.h"

Layer* Layer::setWeightFiller(WeightFiller::Type type)
{
    weightFillerType = type;

    return this;
}