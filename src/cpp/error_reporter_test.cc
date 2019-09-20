
#include "src/cpp/error_reporter.h"

#include "absl/memory/memory.h"
#include "gtest/gtest.h"

namespace coral {
TEST(ErrorReporterTest, CheckReport) {
  EdgeTpuErrorReporter reporter;
  reporter.Report("test");
  EXPECT_EQ("test", reporter.message());

  reporter.Report("test %d", 2);
  EXPECT_EQ("test 2", reporter.message());

  reporter.Report("test %d %s", 3, "test");
  EXPECT_EQ("test 3 test", reporter.message());
}

TEST(ErrorReporterTest, CheckEmptyMessage) {
  EdgeTpuErrorReporter reporter;
  EXPECT_EQ("", reporter.message());
}

}  // namespace coral
