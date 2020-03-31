#include "absl/flags/parse.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "src/cpp/test_utils.h"

namespace coral {
namespace {

TEST(DetectionEngineTest, TestSSDModelsWithCat) {
  // Mobilenet V1 SSD.
  // 4 tensors are returned after post processing operator.
  //
  // 1: detected bounding boxes;
  // 2: detected class label;
  // 3: detected score;
  // 4: number of detected objects;
  TestCatMsCocoDetection(
      TestDataPath("ssd_mobilenet_v1_coco_quant_postprocess.tflite"),
      /*score_threshold=*/0.79, /*iou_threshold=*/0.8);
  TestCatMsCocoDetection(
      TestDataPath("ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite"),
      /*score_threshold=*/0.79, /*iou_threshold=*/0.8);

  // Mobilenet V2 SSD
  TestCatMsCocoDetection(
      TestDataPath("ssd_mobilenet_v2_coco_quant_postprocess.tflite"),
      /*score_threshold=*/0.95, /*iou_threshold=*/0.86);
  TestCatMsCocoDetection(
      TestDataPath("ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"),
      /*score_threshold=*/0.96, /*iou_threshold=*/0.86);
}

void TestFaceDetection(const std::string& model_name, float score_threshold,
                       float iou_threshold) {
  TestDetection(TestDataPath(model_name), TestDataPath("grace_hopper.bmp"),
                /*expected_box=*/{0.29, 0.21, 0.74, 0.57}, /*expected_label=*/0,
                score_threshold, iou_threshold);
}

TEST(DetectionEngineTest, TestFaceModel) {
  TestFaceDetection("ssd_mobilenet_v2_face_quant_postprocess.tflite", 0.7,
                    0.65);
  TestFaceDetection("ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite",
                    0.7, 0.65);
}

TEST(DetectionEngineTest, TestFineTunedPetModel) {
  TestDetection(TestDataPath("ssd_mobilenet_v1_fine_tuned_pet.tflite"),
                TestDataPath("cat.bmp"),
                /*expected_box=*/{0.35, 0.11, 0.7, 0.66},
                /*expected_label=*/0, /*score_threshold=*/0.9,
                /*iou_threshold=*/0.81);

  TestDetection(TestDataPath("ssd_mobilenet_v1_fine_tuned_pet_edgetpu.tflite"),
                TestDataPath("cat.bmp"),
                /*expected_box=*/{0.35, 0.11, 0.7, 0.66},
                /*expected_label=*/0, /*score_threshold=*/0.9,
                /*iou_threshold=*/0.81);
}

}  // namespace
}  // namespace coral

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
