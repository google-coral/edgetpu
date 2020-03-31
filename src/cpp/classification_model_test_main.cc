#include <cstddef>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "src/cpp/classification/engine.h"
#include "src/cpp/test_utils.h"
#include "src/cpp/utils.h"

ABSL_FLAG(std::string, test_case_csv, "",
          "A csv file list all test cases. Path is relative to "
          "test_data folder. Each line has the format: "
          "‘model_path,image_path,effective_scale,effective_means,rgb2bgr,"
          "score_threshold,k,expected_topk_label’. "
          "‘model_path’ and ‘image_path’ are relative to test_data folder."
          "‘effective_means’ should be a colon separated list.");

// If --model_path is set, it will ignore --test_case_csv, and
// only test one model. The test parameters are specified by the following
// flags.
ABSL_FLAG(std::string, model_path, "",
          "Model path relative to test_data folder.");
ABSL_FLAG(std::string, image_path, "cat.bmp",
          "Model path relative to test_data folder.");
ABSL_FLAG(float, effective_scale, 1, "Effective scale of input tensor.");
ABSL_FLAG(std::string, effective_means, "0:0:0",
          "Effective means of input tensor.");
ABSL_FLAG(bool, rgb2bgr, false,
          "Whether convert input image space from RGB to BGR.");
ABSL_FLAG(float, score_threshold, 0, "Threshold on classification score.");
ABSL_FLAG(int, k, 3, "Top k results.");
ABSL_FLAG(int, expected_topk_label, -1,
          "Label expected to seen among top k results");

namespace coral {
namespace {

constexpr size_t kNumCsvFields = 8;

// Parse effective mean values from string.
std::vector<float> ParseEffectiveMeansFromString(const std::string& means_str) {
  std::vector<float> means;
  means.reserve(3);
  const std::vector<std::string> mean_fields = absl::StrSplit(means_str, ':');
  CHECK_EQ(mean_fields.size(), 3);
  for (const auto& mean_str : mean_fields) {
    float mean_v;
    CHECK(absl::SimpleAtof(mean_str, &mean_v));
    means.push_back(mean_v);
  }
  return means;
}

// A csv line with the format:
// model_path,image_path,effective_scale,effective_means,rgb2bgr,
// score_threshold,k,expected_topk_label
ClassificationTestParams ParseTestCaseFromLine(const std::string& csv_line) {
  const std::vector<std::string> fields = absl::StrSplit(csv_line, ',');
  CHECK_EQ(fields.size(), kNumCsvFields);
  ClassificationTestParams params;
  params.model_path = TestDataPath(fields[0]);
  params.image_path = TestDataPath(fields[1]);
  CHECK(absl::SimpleAtof(fields[2], &params.effective_scale));
  if (!fields[3].empty()) {
    params.effective_means = ParseEffectiveMeansFromString(fields[3]);
  }
  params.rgb2bgr = (fields[4] == "true") || (fields[4] == "True");
  CHECK(absl::SimpleAtof(fields[5], &params.score_threshold));
  CHECK(absl::SimpleAtoi(fields[6], &params.k));
  CHECK(absl::SimpleAtoi(fields[7], &params.expected_topk_label));
  return params;
}

std::vector<ClassificationTestParams> ParseTestCasesFromFlags() {
  std::vector<ClassificationTestParams> cases;
  if (!absl::GetFlag(FLAGS_model_path).empty()) {
    ClassificationTestParams params;
    params.model_path = TestDataPath(absl::GetFlag(FLAGS_model_path));
    params.image_path = TestDataPath(absl::GetFlag(FLAGS_image_path));
    params.effective_scale = absl::GetFlag(FLAGS_effective_scale);
    params.effective_means =
        ParseEffectiveMeansFromString(absl::GetFlag(FLAGS_effective_means));
    params.rgb2bgr = absl::GetFlag(FLAGS_rgb2bgr);
    params.score_threshold = absl::GetFlag(FLAGS_score_threshold);
    params.k = absl::GetFlag(FLAGS_k);
    params.expected_topk_label = absl::GetFlag(FLAGS_expected_topk_label);
    cases.push_back(params);
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

TEST(ClassificationModelTest, Top3ContainsLabel) {
  const auto& test_cases = ParseTestCasesFromFlags();
  ASSERT_TRUE(!test_cases.empty());
  for (const auto& case_params : test_cases) {
    TestClassification(case_params);
  }
}

}  // namespace
}  // namespace coral

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
