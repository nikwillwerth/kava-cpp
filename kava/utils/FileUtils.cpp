#include <fstream>
#include <iostream>
#include "FileUtils.h"

bool FileUtils::doesFileExist(const std::string filename)
{
    std::ifstream infile(filename);
    return infile.good();
}