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

    kernelArea   = kernelSize  * kernelSize;
    kernelLength = kernelArea  * bottomBlobs[0]->channels;
    outputLength = outputWidth * outputHeight;

    im2colMatrix = MatrixXf(outputLength, kernelLength);

    topBlobs[0]->reshape(numOutputs, outputHeight, outputWidth);
    weightBlobs[0]->reshape(1, kernelLength, numOutputs);

    WeightFiller::getWeightFillerWithType(weightFillerType)->fill(weightBlobs[0], kernelLength, numOutputs);
}

void ConvolutionalLayer::forward()
{
    //im2col(I)
    int rowIndex = 0;

    for(int c = 0; c < outputWidth; c++)
    {
        int col = (c * stride);

        for(int r = 0; r < outputHeight; r++)
        {
            int row = (r * stride);

            for(int channel = 0; channel < bottomBlobs[0]->channels; channel++)
            {
                int colIndex = (col + (channel * bottomBlobs[0]->width));

                MatrixXf block = bottomBlobs[0]->dataMatrix.block(row, colIndex, kernelSize, kernelSize).matrix();
                block.resize(1, kernelArea);

                int startColumn = channel * kernelArea;

                im2colMatrix.block(rowIndex, startColumn, 1, kernelArea) = block;
            }

            rowIndex++;
        }
    }

    //im2col(I) * W
    MatrixXf result = im2colMatrix * weightBlobs[0]->dataMatrix;

    topBlobs[0]->dataMatrix.resize(outputHeight, outputWidth * numOutputs);

    for(int i = 0; i < numOutputs; i++)
    {
        MatrixXf block = result.col(i).matrix();
        block.resize(outputHeight, outputWidth);

        topBlobs[0]->dataMatrix.block(0, i * outputWidth, outputHeight, outputWidth) = block;
    }
}

void ConvolutionalLayer::backward()
{
    /*int r = topBlobs[0]->diffMatrix.rows();
    int c = topBlobs[0]->diffMatrix.cols();

    MatrixXf dY = MatrixXf(r + 2, c + (2 * numOutputs));

    std::cout << dY << std::endl;

    for(int i = 0; i < numOutputs; i++)
    {
        dY.block(1, i * c, r, c) = topBlobs[0]->diffMatrix.block(0, i * c, r, c);
    }

    std::cout << dY << std::endl;*/

    /*//topBlobs[0]->dataMatrix.resize(outputHeight, outputWidth * numOutputs);
    //topBlobs[0]->diffMatrix.resize(numOutputs, outputLength);

    //std::cout << topBlobs[0]->dataMatrix << std::endl << std::endl;
    std::cout << topBlobs[0]->diffMatrix << std::endl << std::endl;

    MatrixXf temp = MatrixXf(outputLength, numOutputs);

    for(int i = 0; i < numOutputs; i++)
    {
        MatrixXf block = topBlobs[0]->diffMatrix.block((i * outputLength), 0, outputLength, 1).matrix();
        block.resize(outputLength, 1);

        temp.col(i) = block;
    }

    //std::cout << temp << std::endl << std::endl;

    //std::cout << im2colMatrix.rows() << "x" << im2colMatrix.cols() << std::endl;
    //std::cout << topBlobs[0]->diffMatrix.rows() << "x" << topBlobs[0]->diffMatrix.cols() << std::endl;

    //dW
    weightBlobs[0]->diffMatrix = im2colMatrix * temp;

    //std::cout << weightBlobs[0]->diffMatrix << std::endl << std::endl;
    //std::cout << weightBlobs[0]->dataMatrix << std::endl << std::endl;*/
}

ConvolutionalLayer* ConvolutionalLayer::setKernelSize(int kernelSize)
{
    this->kernelSize = kernelSize;

    return this;
}

ConvolutionalLayer* ConvolutionalLayer::setStride(int stride)
{
    this->stride = stride;

    return this;
}

ConvolutionalLayer* ConvolutionalLayer::setPadding(int padding)
{
    this->padding = padding;

    return this;
}