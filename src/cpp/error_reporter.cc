#include "src/cpp/error_reporter.h"

namespace coral {

void EdgeTpuErrorReporter::Report(const std::string& msg) { buffer_ << msg; }

// Reports an error message with args.
int EdgeTpuErrorReporter::Report(const char* format, va_list args) {
  char buf[1024];
  int formatted = vsnprintf(buf, sizeof(buf), format, args);
  buffer_ << buf;
  return formatted;
}

// Gets the last error message and clears the buffer.
std::string EdgeTpuErrorReporter::message() {
  std::string value = buffer_.str();
  // clear() for flag status.
  buffer_.clear();
  // clear string content.
  buffer_.str("");
  return value;
}

}  // namespace coral
