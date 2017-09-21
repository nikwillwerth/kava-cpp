#include "ConstantWeightFiller.h"

void ConstantWeightFiller::fill(Blob *blob, int numInputs, int numOutputs)
{
    for(int i = 0; i < blob->count; i++)
    {
        blob->data[i] = 1;
    }

    new (&blob->dataMatrix) Map<MatrixXf>(blob->data, blob->height, blob->width);
}