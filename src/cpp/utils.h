#ifndef EDGETPU_CPP_UTILS_H_
#define EDGETPU_CPP_UTILS_H_

#include <string>

#include "src/cpp/error_reporter.h"

namespace coral {

// Reads contents of file.
EdgeTpuApiStatus ReadFile(const std::string& file_path,
                          std::string* file_content,
                          EdgeTpuErrorReporter* reporter);

// Writes contents of file.
EdgeTpuApiStatus WriteFile(const std::string& file_content,
                           const std::string& file_path,
                           EdgeTpuErrorReporter* reporter);

// Reads contents of file or crashes if error.
void ReadFileOrDie(const std::string& file_path, std::string* file_content);

// Writes contents of file or crashes if error.
void WriteFileOrDie(const std::string& file_content,
                    const std::string& file_path);

}  // namespace coral
#endif  // EDGETPU_CPP_UTILS_H_
