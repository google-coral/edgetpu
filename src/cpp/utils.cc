#include "src/cpp/utils.h"

#include <sys/stat.h>

#include <memory>

#include "absl/strings/str_cat.h"
#include "glog/logging.h"

namespace coral {

EdgeTpuApiStatus ReadFile(const std::string& file_path,
                          std::string* file_content,
                          EdgeTpuErrorReporter* reporter) {
  CHECK(file_content);
  CHECK(reporter);
  std::unique_ptr<FILE, decltype(&std::fclose)> file(
      std::fopen(file_path.c_str(), "rb"), std::fclose);
  if (!file) {
    reporter->Report(absl::StrCat("Failed to open file: ", file_path));
    return kEdgeTpuApiError;
  }
  struct stat file_stat;
  if (fstat(fileno(file.get()), &file_stat) != 0) {
    reporter->Report(
        absl::StrCat("Failed to get the size of file: ", file_path));
    return kEdgeTpuApiError;
  }

  const auto file_size_bytes = file_stat.st_size;
  file_content->resize(file_size_bytes);
  size_t bytes_read = std::fread(&(*file_content->begin()), sizeof(char),
                                 file_size_bytes, file.get());
  if (bytes_read != file_size_bytes) {
    reporter->Report(
        absl::StrCat("Failed to read ", file_size_bytes, " bytes of data!"));
    return kEdgeTpuApiError;
  }
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus WriteFile(const std::string& file_content,
                           const std::string& file_path,
                           EdgeTpuErrorReporter* reporter) {
  std::unique_ptr<FILE, decltype(&std::fclose)> file(
      std::fopen(file_path.c_str(), "wb"), std::fclose);
  if (!file) {
    reporter->Report(absl::StrCat("Failed to open file: ", file_path));
    return kEdgeTpuApiError;
  }

  size_t bytes_written = std::fwrite(file_content.data(), sizeof(char),
                                     file_content.size(), file.get());
  if (bytes_written != file_content.size()) {
    reporter->Report(absl::StrCat("Failed to write ", file_content.size(),
                                  " bytes of data!"));
    return kEdgeTpuApiError;
  }
  return kEdgeTpuApiOk;
}

void ReadFileOrDie(const std::string& file_path, std::string* file_content) {
  CHECK(file_content);
  EdgeTpuErrorReporter reporter;
  CHECK_EQ(ReadFile(file_path, file_content, &reporter), kEdgeTpuApiOk)
      << "Unable to open file: " << file_path;
}

void WriteFileOrDie(const std::string& file_content,
                    const std::string& file_path) {
  EdgeTpuErrorReporter reporter;
  CHECK_EQ(WriteFile(file_content, file_path, &reporter), kEdgeTpuApiOk);
}

}  // namespace coral
