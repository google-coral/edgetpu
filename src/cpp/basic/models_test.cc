// Tests correctness of models for tasks other than classification and
// detection. The classification and detection models would be tested by test
// suite under their engine folders.

#include <cmath>
#include <iostream>

#include "absl/flags/parse.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "src/cpp/basic/basic_engine.h"
#include "src/cpp/test_utils.h"

namespace coral {

// This tests the correctness of self trained classification model which follows
// `Imprinted Weights` transfer learning method proposed in paper
// https://arxiv.org/pdf/1712.07136.pdf.
TEST(ModelCorrectnessTest, TestMobilenetV1WithL2Norm) {
  BasicEngine engine(TestDataPath("mobilenet_v1_1.0_224_l2norm_quant.tflite"));
  // Tests with cat and bird.
  std::vector<uint8_t> cat_input =
      GetInputFromImage(TestDataPath("cat.bmp"), {224, 224, 3});
  std::vector<uint8_t> bird_input =
      GetInputFromImage(TestDataPath("bird.bmp"), {224, 224, 3});
  auto results = engine.RunInference(cat_input);
  ASSERT_EQ(1, results.size());
  auto result = results[0];
  int class_max = std::distance(result.begin(),
                                std::max_element(result.begin(), result.end()));
  EXPECT_EQ(class_max, 286);
  EXPECT_GT(result[286], 0.68);  // Egyptian cat

  results = engine.RunInference(bird_input);
  ASSERT_EQ(1, results.size());
  result = results[0];
  class_max = std::distance(result.begin(),
                            std::max_element(result.begin(), result.end()));
  EXPECT_EQ(class_max, 20);
  EXPECT_GT(result[20], 0.97);  // chickadee
}

// TODO: move co-compile tests into a separate test file.
TEST(ModelCorrectnessTest, TestCocompiledModels) {
  // Mobilenet V1 0.25 and Mobilenet V1 0.5.
  TestClassification(
      TestDataPath("cocompilation/mobilenet_v1_0.25_128_quant_cocompiled_with_"
                   "mobilenet_v1_0.5_160_quant_edgetpu.tflite"),
      TestDataPath("cat.bmp"),
      /*score_threshold=*/0.36,
      /*expected_top1_label=*/283);  // tiger cat
  TestClassification(
      TestDataPath("cocompilation/mobilenet_v1_0.5_160_quant_cocompiled_with_"
                   "mobilenet_v1_0.25_128_quant_edgetpu.tflite"),
      TestDataPath("cat.bmp"),
      /*score_threshold=*/0.68,
      /*expected_top1_label=*/286);  // Egyptian cat

  // Inception V3 and Inception V4.
  TestClassification(
      TestDataPath("cocompilation/inception_v3_299_quant_cocompiled_with_"
                   "inception_v4_299_quant_edgetpu.tflite"),
      TestDataPath("cat.bmp"),
      /*score_threshold=*/0.597,
      /*expected_top1_label=*/282);  // tabby, tabby cat
  TestClassification(
      TestDataPath("cocompilation/inception_v4_299_quant_cocompiled_with_"
                   "inception_v3_299_quant_edgetpu.tflite"),
      TestDataPath("cat.bmp"),
      /*score_threshold=*/0.41,
      /*expected_top1_label=*/282);  // tabby, tabby cat

  // Mobilenet V1 0.25 and Inception V4.
  TestClassification(
      TestDataPath("cocompilation/mobilenet_v1_0.25_128_quant_cocompiled_with_"
                   "inception_v4_299_quant_edgetpu.tflite"),
      TestDataPath("cat.bmp"),
      /*score_threshold=*/0.36,
      /*expected_top1_label=*/283);  // tiger cat
  TestClassification(
      TestDataPath("cocompilation/inception_v4_299_quant_cocompiled_with_"
                   "mobilenet_v1_0.25_128_quant_edgetpu.tflite"),
      TestDataPath("cat.bmp"),
      /*score_threshold=*/0.41,
      /*expected_top1_label=*/282);  // tabby, tabby cat

  // Mobilenet V1 1.0, Mobilenet V1 0.25, Mobilenet V1 0.5, Mobilenet V1 0.75.
  TestClassification(
      TestDataPath(
          "cocompilation/"
          "mobilenet_v1_1.0_224_quant_cocompiled_with_3quant_edgetpu.tflite"),
      TestDataPath("cat.bmp"),
      /*score_threshold=*/0.78,
      /*expected_top1_label=*/286);  // Egyptian cat
  TestClassification(
      TestDataPath(
          "cocompilation/"
          "mobilenet_v1_0.25_128_quant_cocompiled_with_3quant_edgetpu.tflite"),
      TestDataPath("cat.bmp"),
      /*score_threshold=*/0.36,
      /*expected_top1_label=*/283);  // tiger cat
  TestClassification(
      TestDataPath(
          "cocompilation/"
          "mobilenet_v1_0.5_160_quant_cocompiled_with_3quant_edgetpu.tflite"),
      TestDataPath("cat.bmp"),
      /*score_threshold=*/0.68,
      /*expected_top1_label=*/286);  // Egyptian cat
  TestClassification(
      TestDataPath(
          "cocompilation/"
          "mobilenet_v1_0.75_192_quant_cocompiled_with_3quant_edgetpu.tflite"),
      TestDataPath("cat.bmp"),
      /*score_threshold=*/0.4,
      /*expected_top1_label=*/283);  // tiger cat

  // Inception V1, Inception V2, Inception V3 and Inception V4.
  TestClassification(
      TestDataPath(
          "cocompilation/"
          "inception_v1_224_quant_cocompiled_with_3quant_edgetpu.tflite"),
      TestDataPath("cat.bmp"),
      /*score_threshold=*/0.38,
      /*expected_top1_label=*/286);  // Egyptian cat
  TestClassification(
      TestDataPath(
          "cocompilation/"
          "inception_v2_224_quant_cocompiled_with_3quant_edgetpu.tflite"),
      TestDataPath("cat.bmp"),
      /*score_threshold=*/0.61,
      /*expected_top1_label=*/286);  // Egyptian cat
  TestClassification(
      TestDataPath(
          "cocompilation/"
          "inception_v3_299_quant_cocompiled_with_3quant_edgetpu.tflite"),
      TestDataPath("cat.bmp"),
      /*score_threshold=*/0.597,
      /*expected_top1_label=*/282);  // tabby, tabby cat
  TestClassification(
      TestDataPath(
          "cocompilation/"
          "inception_v4_299_quant_cocompiled_with_3quant_edgetpu.tflite"),
      TestDataPath("cat.bmp"),
      /*score_threshold=*/0.41,
      /*expected_top1_label=*/282);  // tabby, tabby cat
}

TEST(ModelCorrectnessTest, Deeplab513Mv2Dm1_WithArgMax) {
  // See label map:
  // test_data/pascal_voc_segmentation_labels.txt
  std::vector<uint8_t> cpu_pred_segmentation;
  TestSegmentationWithArgmax("deeplabv3_mnv2_pascal_quant.tflite",
                             "bird_segmentation.bmp",
                             "bird_segmentation_mask.bmp",
                             /*size=*/513,
                             /*iou_threshold=*/0.95, &cpu_pred_segmentation);
  std::vector<uint8_t> edgetpu_pred_segmentation;
  TestSegmentationWithArgmax(
      "deeplabv3_mnv2_pascal_quant_edgetpu.tflite", "bird_segmentation.bmp",
      "bird_segmentation_mask.bmp",
      /*size=*/513,
      /*iou_threshold=*/0.95, &edgetpu_pred_segmentation);
  float seg_iou = ComputeIntersectionOverUnion(cpu_pred_segmentation,
                                               edgetpu_pred_segmentation);
  EXPECT_GT(seg_iou, 0.99);
}

TEST(ModelCorrectnessTest, Deeplab513Mv2Dm05_WithArgMax) {
  // See label map:
  // test_data/pascal_voc_segmentation_labels.txt
  std::vector<uint8_t> cpu_pred_segmentation;
  TestSegmentationWithArgmax("deeplabv3_mnv2_dm05_pascal_quant.tflite",
                             "bird_segmentation.bmp",
                             "bird_segmentation_mask.bmp",
                             /*size=*/513,
                             /*iou_threshold=*/0.94, &cpu_pred_segmentation);
  std::vector<uint8_t> edgetpu_pred_segmentation;
  TestSegmentationWithArgmax(
      "deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite",
      "bird_segmentation.bmp", "bird_segmentation_mask.bmp",
      /*size=*/513,
      /*iou_threshold=*/0.94, &edgetpu_pred_segmentation);
  float seg_iou = ComputeIntersectionOverUnion(cpu_pred_segmentation,
                                               edgetpu_pred_segmentation);
  EXPECT_GT(seg_iou, 0.98);
}

}  // namespace coral

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
