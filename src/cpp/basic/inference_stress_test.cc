#include <cmath>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "src/cpp/test_utils.h"

ABSL_FLAG(int, stress_test_runs, 500, "Number of iterations for stress test.");
ABSL_FLAG(int, stress_with_sleep_test_runs, 200,
          "Number of iterations for stress test.");
ABSL_FLAG(int, stress_sleep_sec, 3,
          "Seconds to sleep in-between inference runs.");

namespace coral {
namespace {

TEST(InferenceStressTest, MobilenetV1) {
  InferenceStressTest(TestDataPath("mobilenet_v1_1.0_224_quant_edgetpu.tflite"),
                      absl::GetFlag(FLAGS_stress_test_runs));
}

TEST(InferenceStressTest, MobilenetV1SSD) {
  InferenceStressTest(
      TestDataPath("mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite"),
      absl::GetFlag(FLAGS_stress_test_runs));
}

TEST(InferenceStressTest, InceptionV2) {
  InferenceStressTest(TestDataPath("inception_v2_224_quant_edgetpu.tflite"),
                      absl::GetFlag(FLAGS_stress_test_runs));
}

TEST(InferenceStressTest, InceptionV4) {
  InferenceStressTest(TestDataPath("inception_v4_299_quant_edgetpu.tflite"),
                      absl::GetFlag(FLAGS_stress_test_runs));
}

// Stress tests with sleep in-between inference runs.
// We cap the runs here as they will take a lot of time to finish.
TEST(InferenceStressTest, MobilenetV1_WithSleep) {
  InferenceStressTest(TestDataPath("mobilenet_v1_1.0_224_quant_edgetpu.tflite"),
                      absl::GetFlag(FLAGS_stress_with_sleep_test_runs),
                      absl::GetFlag(FLAGS_stress_sleep_sec));
}

TEST(InferenceStressTest, InceptionV2_WithSleep) {
  InferenceStressTest(TestDataPath("inception_v2_224_quant_edgetpu.tflite"),
                      absl::GetFlag(FLAGS_stress_with_sleep_test_runs),
                      absl::GetFlag(FLAGS_stress_sleep_sec));
}

}  // namespace
}  // namespace coral

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
