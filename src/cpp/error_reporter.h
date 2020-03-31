#ifndef EDGETPU_CPP_ERROR_REPORTER_H_
#define EDGETPU_CPP_ERROR_REPORTER_H_

#include <sstream>
#include <string>

#include "tensorflow/lite/core/api/error_reporter.h"

namespace coral {

typedef enum { kEdgeTpuApiOk = 0, kEdgeTpuApiError = 1 } EdgeTpuApiStatus;

#define EDGETPU_API_ENSURE_STATUS(status) \
  do {                                    \
    if ((status) != kEdgeTpuApiOk) {      \
      return kEdgeTpuApiError;            \
    }                                     \
  } while (0)

#define EDGETPU_API_ENSURE(value) \
  do {                            \
    if (!(value)) {               \
      return kEdgeTpuApiError;    \
    }                             \
  } while (0)

#define EDGETPU_API_REPORT_ERROR(reporter, condition, msg) \
  do {                                                     \
    if (condition) {                                       \
      (reporter)->Report(msg);                             \
      return kEdgeTpuApiError;                             \
    }                                                      \
  } while (0)

#define EDGETPU_API_REPORT_ERROR_WITH_ARGS(reporter, condition, msg, ...) \
  do {                                                                    \
    if (condition) {                                                      \
      (reporter)->Report(msg, __VA_ARGS__);                               \
      return kEdgeTpuApiError;                                            \
    }                                                                     \
  } while (0)

class EdgeTpuErrorReporter : public tflite::ErrorReporter {
 public:
  void Report(const std::string& msg);

  // We declared two functions with name 'Report', so that the variadic Report
  // function in tflite::ErrorReporter is hidden.
  // See https://isocpp.org/wiki/faq/strange-inheritance#hiding-rule.
  using tflite::ErrorReporter::Report;

  // Reports an error message with args.
  int Report(const char* format, va_list args) override;

  // Gets the last error message and clears the buffer.
  std::string message();

 private:
  std::stringstream buffer_;
};

}  // namespace coral

#endif  // EDGETPU_CPP_ERROR_REPORTER_H_
