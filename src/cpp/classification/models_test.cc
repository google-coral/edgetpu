#include "absl/flags/parse.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "src/cpp/classification/engine.h"
#include "src/cpp/test_utils.h"

namespace coral {
namespace {

TEST(ClassificationEngineTest, TestMobilenetModels) {
  // Mobilenet V1 1.0
  TestClassification(TestDataPath("mobilenet_v1_1.0_224_quant.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.78,
                     /*expected_top1_label=*/286);  // Egyptian cat
  TestClassification(TestDataPath("mobilenet_v1_1.0_224_quant_edgetpu.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.78,
                     /*expected_top1_label=*/286);  // Egyptian cat

  // Mobilenet V1 0.25
  TestClassification(TestDataPath("mobilenet_v1_0.25_128_quant.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.36,
                     /*expected_top1_label=*/283);  // tiger cat
  TestClassification(TestDataPath("mobilenet_v1_0.25_128_quant_edgetpu.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.36,
                     /*expected_top1_label=*/283);  // tiger cat

  // Mobilenet V1 0.5
  TestClassification(TestDataPath("mobilenet_v1_0.5_160_quant.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.68,
                     /*expected_top1_label=*/286);  // Egyptian cat
  TestClassification(TestDataPath("mobilenet_v1_0.5_160_quant_edgetpu.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.68,
                     /*expected_top1_label=*/286);  // Egyptian cat
  // Mobilenet V1 0.75
  TestClassification(TestDataPath("mobilenet_v1_0.75_192_quant.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.4,
                     /*expected_top1_label=*/283);  // tiger cat
  TestClassification(TestDataPath("mobilenet_v1_0.75_192_quant_edgetpu.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.4,
                     /*expected_top1_label=*/283);  // tiger cat

  // Mobilenet V2
  TestClassification(TestDataPath("mobilenet_v2_1.0_224_quant.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.8,
                     /*expected_top1_label=*/286);  // Egyptian cat
  TestClassification(TestDataPath("mobilenet_v2_1.0_224_quant_edgetpu.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.79,
                     /*expected_top1_label=*/286);  // Egyptian cat
}

TEST(ClassificationEngineTest, TestInceptionModels) {
  // Inception V1
  TestClassification(TestDataPath("inception_v1_224_quant.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.37,
                     /*expected_top1_label=*/282);  // tabby, tabby cat

  TestClassification(TestDataPath("inception_v1_224_quant_edgetpu.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.38,
                     /*expected_top1_label=*/286);  // Egyptian cat

  // Inception V2
  TestClassification(TestDataPath("inception_v2_224_quant.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.65,
                     /*expected_top1_label=*/286);  // Egyptian cat
  TestClassification(TestDataPath("inception_v2_224_quant_edgetpu.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.61,
                     /*expected_top1_label=*/286);  // Egyptian cat

  // Inception V3
  TestClassification(TestDataPath("inception_v3_299_quant.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.58,
                     /*expected_top1_label=*/282);  // tabby, tabby cat
  TestClassification(TestDataPath("inception_v3_299_quant_edgetpu.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.597,
                     /*expected_top1_label=*/282);  // tabby, tabby cat

  // Inception V4
  TestClassification(TestDataPath("inception_v4_299_quant.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.35,
                     /*expected_top1_label=*/286);  // Egyptian cat
  TestClassification(TestDataPath("inception_v4_299_quant_edgetpu.tflite"),
                     TestDataPath("cat.bmp"),
                     /*score_threshold=*/0.41,
                     /*expected_top1_label=*/282);  // tabby, tabby cat
}

TEST(ClassificationEngineTest, TestINatModels) {
  // Plant model
  TestClassification(
      TestDataPath("mobilenet_v2_1.0_224_inat_plant_quant.tflite"),
      TestDataPath("sunflower.bmp"),
      /*score_threshold=*/0.8,
      /*expected_top1_label=*/1680);  // Helianthus annuus (common sunflower)
  TestClassification(
      TestDataPath("mobilenet_v2_1.0_224_inat_plant_quant_edgetpu.tflite"),
      TestDataPath("sunflower.bmp"),
      /*score_threshold=*/0.8,
      /*expected_top1_label=*/1680);  // Helianthus annuus (common sunflower)

  // Insect model
  TestClassification(
      TestDataPath("mobilenet_v2_1.0_224_inat_insect_quant.tflite"),
      TestDataPath("dragonfly.bmp"), /*score_threshold=*/0.2,
      /*expected_top1_label=*/912);  // Thornbush Dasher
  TestClassification(
      TestDataPath("mobilenet_v2_1.0_224_inat_insect_quant_edgetpu.tflite"),
      TestDataPath("dragonfly.bmp"), /*score_threshold=*/0.2,
      /*expected_top1_label=*/912);  // Thornbush Dasher

  // Bird model
  TestClassification(
      TestDataPath("mobilenet_v2_1.0_224_inat_bird_quant.tflite"),
      TestDataPath("bird.bmp"), /*score_threshold=*/0.5,
      /*expected_top1_label=*/659);  // Black-capped Chickadee

  TestClassification(
      TestDataPath("mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite"),
      TestDataPath("bird.bmp"), /*score_threshold=*/0.5,
      /*expected_top1_label=*/659);  // Black-capped Chickadee
}

TEST(ClassificationEngineTest,
     TestEfficientNetEdgeTpuModelsCustomPreprocessing) {
  const int kTopk = 3;
  // Custom preprocessing is done by:
  // (v - (mean - zero_point * scale * stddev)) / (stddev * scale)
  {
    // mean 127, stddev 128
    // first input tensor scale: 0.012566, zero_point: 131
    const float effective_scale = 128 * 0.012566;
    const std::vector<float> effective_means(3, 127 - 131 * effective_scale);
    TestClassification(TestDataPath("efficientnet-edgetpu-S_quant.tflite"),
                       TestDataPath("cat.bmp"), effective_scale,
                       effective_means,
                       /*score_threshold=*/0.4, kTopk,
                       /*expected_topk_label=*/286);  // Egyptian cat
    TestClassification(
        TestDataPath("efficientnet-edgetpu-S_quant_edgetpu.tflite"),
        TestDataPath("cat.bmp"), effective_scale, effective_means,
        /*score_threshold=*/0.4, kTopk,
        /*expected_topk_label=*/286);  // Egyptian cat
  }

  {
    // mean 127, stddev 128
    // first input tensor scale: 0.012089, zero_point: 129
    const float effective_scale = 128 * 0.012089;
    const std::vector<float> effective_means(3, 127 - 129 * effective_scale);
    TestClassification(TestDataPath("efficientnet-edgetpu-M_quant.tflite"),
                       TestDataPath("cat.bmp"), effective_scale,
                       effective_means,
                       /*score_threshold=*/0.6, kTopk,
                       /*expected_topk_label=*/286);  // Egyptian cat
    TestClassification(
        TestDataPath("efficientnet-edgetpu-M_quant_edgetpu.tflite"),
        TestDataPath("cat.bmp"), effective_scale, effective_means,
        /*score_threshold=*/0.6, kTopk,
        /*expected_topk_label=*/286);  // Egyptian cat
  }

  {
    // mean 127, stddev 128
    // first input tensor scale: 0.01246, zero_point: 129
    const float effective_scale = 128 * 0.01246;
    const std::vector<float> effective_means(3, 127 - 129 * effective_scale);
    TestClassification(TestDataPath("efficientnet-edgetpu-L_quant.tflite"),
                       TestDataPath("cat.bmp"), effective_scale,
                       effective_means,
                       /*score_threshold=*/0.6, kTopk,
                       /*expected_topk_label=*/286);  // Egyptian cat
    TestClassification(
        TestDataPath("efficientnet-edgetpu-L_quant_edgetpu.tflite"),
        TestDataPath("cat.bmp"), effective_scale, effective_means,
        /*score_threshold=*/0.6, kTopk,
        /*expected_topk_label=*/286);  // Egyptian cat
  }
}
}  // namespace
}  // namespace coral

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
