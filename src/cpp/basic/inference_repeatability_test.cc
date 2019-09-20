#include <cmath>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "src/cpp/test_utils.h"

ABSL_FLAG(int, stress_test_runs, 500, "Number of iterations for stress test.");

namespace coral {
namespace {

TEST(InferenceRepeatabilityTest, MobilenetV1) {
  RepeatabilityTest(TestDataPath("mobilenet_v1_1.0_224_quant_edgetpu.tflite"),
                    absl::GetFlag(FLAGS_stress_test_runs));
}

TEST(InferenceRepeatabilityTest, MobilenetV1SSD) {
  RepeatabilityTest(
      TestDataPath("mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite"),
      absl::GetFlag(FLAGS_stress_test_runs));
}

TEST(InferenceRepeatabilityTest, InceptionV2) {
  RepeatabilityTest(TestDataPath("inception_v2_224_quant_edgetpu.tflite"),
                    absl::GetFlag(FLAGS_stress_test_runs));
}

TEST(InferenceRepeatabilityTest, InceptionV4) {
  RepeatabilityTest(TestDataPath("inception_v4_299_quant_edgetpu.tflite"),
                    absl::GetFlag(FLAGS_stress_test_runs));
}

}  // namespace
}  // namespace coral

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
