#include "ConvolutionalLayer.h"

ConvolutionalLayer::ConvolutionalLayer(const std::string name, std::string bottomBlobName, std::string topBlobName, int numOutputs)
{
    this->name       = name;
    this->numOutputs = numOutputs;

    bottomBlobNames = std::vector<std::string>();
    topBlobs        = std::vector<Blob *>();
    weightBlobs     = std::vector<Blob *>();

    bottomBlobNames.push_back(bottomBlobName);
    topBlobs.push_back(   new Blob(topBlobName));
    weightBlobs.push_back(new Blob("weights"));
}

void ConvolutionalLayer::setUp()
{
    std::cout << "Setting up convolutional layer..." << std::endl;

    outputWidth  = ((bottomBlobs[0]->width  - kernelSize + (2 * padding)) / stride) + 1;
    outputHeight = ((bottomBlobs[0]->height - kernelSize + (2 * padding)) / stride) + 1;

    kernelLength = kernelSize  * kernelSize * bottomBlobs[0]->channels;
    outputLength = outputWidth * outputHeight;

    im2colMatrix = MatrixXf(outputLength, kernelLength);

    topBlobs[0]->reshape(numOutputs, outputHeight, outputWidth);
    weightBlobs[0]->reshape(1, kernelLength, numOutputs);

    WeightFiller::getWeightFillerWithType(WeightFiller::Constant)->fill(weightBlobs[0], kernelLength, numOutputs);
}

void ConvolutionalLayer::forward()
{
    //im2col
    int rowIndex = 0;

    int kernelArea = kernelSize * kernelSize;

    clock_t begin_time = clock();

    for(int r = 0; r < outputHeight; r++)
    {
        for(int c = 0; c < outputWidth; c++)
        {
            for(int channel = 0; channel < bottomBlobs[0]->channels; channel++)
            {
                int row = ( r * stride);
                int col = ((c * stride) + (channel * bottomBlobs[0]->width));

                MatrixXf block = bottomBlobs[0]->dataMatrix.block(row, col, kernelSize, kernelSize).matrix();
                block.resize(1, kernelArea);

                int startColumn = channel * kernelArea;

                im2colMatrix.block(rowIndex, startColumn, 1, kernelArea) = block;
            }

            rowIndex++;
        }
    }

    float numSeconds = float(clock () - begin_time) / CLOCKS_PER_SEC;

    std::cout << "Total time for im2col:     " << numSeconds << std::endl;

    begin_time = clock();

    MatrixXf thisResult = im2colMatrix * weightBlobs[0]->dataMatrix;

    numSeconds = float(clock () - begin_time) / CLOCKS_PER_SEC;

    std::cout << "Total time for GEMM:     " << numSeconds << std::endl;
}

void ConvolutionalLayer::backward()
{

}