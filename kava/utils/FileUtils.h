#ifndef KAVA_CPP_FILEUTILS_H
#define KAVA_CPP_FILEUTILS_H

#include <string>

class FileUtils
{
public:
    static bool doesFileExist(std::string filename);
};

#endif //KAVA_CPP_FILEUTILS_H