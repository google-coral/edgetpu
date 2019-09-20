#include <cmath>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "src/cpp/basic/basic_engine.h"
#include "src/cpp/test_utils.h"

ABSL_FLAG(int, stress_test_runs, 500, "Number of iterations for stress test.");

namespace coral {
namespace {

TEST(ModelLoadingStressTest, AlternateEdgeTpuModels) {
  const std::vector<std::string> model_names = {
      "mobilenet_v1_1.0_224_quant_edgetpu.tflite",
      "mobilenet_v2_1.0_224_quant_edgetpu.tflite",
      "mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite",
      "mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite",
      "inception_v1_224_quant_edgetpu.tflite",
      "inception_v2_224_quant_edgetpu.tflite",
      "inception_v3_299_quant_edgetpu.tflite",
      "inception_v4_299_quant_edgetpu.tflite",
  };

  const int num_runs = absl::GetFlag(FLAGS_stress_test_runs);
  for (int i = 0; i < num_runs; ++i) {
    VLOG_EVERY_N(0, 100) << "Stress test iter " << i << "...";
    for (int j = 0; j < model_names.size(); ++j) {
      BasicEngine engine(TestDataPath(model_names[j]));
    }
  }
}

TEST(ModelLoadingStressTest, AlternateCpuModels) {
  const std::vector<std::string> model_names = {
      "mobilenet_v1_1.0_224_quant.tflite",
      "mobilenet_v2_1.0_224_quant.tflite",
      "mobilenet_ssd_v1_coco_quant_postprocess.tflite",
      "mobilenet_ssd_v2_coco_quant_postprocess.tflite",
      "inception_v1_224_quant.tflite",
      "inception_v2_224_quant.tflite",
      "inception_v3_299_quant.tflite",
      "inception_v4_299_quant.tflite",
  };

  const int num_runs = absl::GetFlag(FLAGS_stress_test_runs);
  for (int i = 0; i < num_runs; ++i) {
    VLOG_EVERY_N(0, 100) << "Stress test iter " << i << "...";
    for (int j = 0; j < model_names.size(); ++j) {
      BasicEngine engine(TestDataPath(model_names[j]));
    }
  }
}

}  // namespace
}  // namespace coral

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
