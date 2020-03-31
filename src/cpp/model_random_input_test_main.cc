#include <cstddef>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "src/cpp/test_utils.h"
#include "src/cpp/utils.h"

ABSL_FLAG(std::string, test_case_csv, "",
          "A csv file list all test cases. Path is relative to "
          "test_data folder. Each line has the format: "
          "‘model_path,expected_num_output_tensors’. "
          "‘model_path’ is relative to test_data folder.");

// If --model_path is set, it will ignore --test_case_csv, and
// only test one model.
ABSL_FLAG(std::string, model_path, "",
          "Model path relative to test_data folder.");
ABSL_FLAG(int, expected_num_output_tensors, 1,
          "Expected number of output tensors");

namespace coral {
namespace {

constexpr size_t kNumCsvFields = 2;

// A csv line with the format:
// model_path,expected_num_output_tensors
RandomInputTestParams ParseTestCaseFromLine(const std::string& csv_line) {
  const std::vector<std::string> fields = absl::StrSplit(csv_line, ',');
  CHECK_EQ(fields.size(), kNumCsvFields);
  RandomInputTestParams params;
  params.model_path = TestDataPath(fields[0]);
  CHECK(absl::SimpleAtoi(fields[1], &params.expected_num_output_tensors));
  return params;
}

std::vector<RandomInputTestParams> ParseTestCasesFromFlags() {
  std::vector<RandomInputTestParams> cases;
  if (!absl::GetFlag(FLAGS_model_path).empty()) {
    cases.push_back({absl::GetFlag(FLAGS_model_path),
                     absl::GetFlag(FLAGS_expected_num_output_tensors)});
  } else {
    CHECK(!absl::GetFlag(FLAGS_test_case_csv).empty());
    std::string file_content;
    ReadFileOrDie(TestDataPath(absl::GetFlag(FLAGS_test_case_csv)),
                  &file_content);
    const std::vector<std::string> lines =
        absl::StrSplit(file_content, '\n', absl::SkipWhitespace());
    // Skip the CSV header line.
    for (int i = 1; i < lines.size(); ++i) {
      cases.push_back(ParseTestCaseFromLine(lines[i]));
    }
  }
  return cases;
}

TEST(ModelRandomInputTest, RunInference) {
  const auto& test_cases = ParseTestCasesFromFlags();
  ASSERT_TRUE(!test_cases.empty());
  for (const auto& case_params : test_cases) {
    TestWithRandomInput(case_params);
  }
}

}  // namespace
}  // namespace coral

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
