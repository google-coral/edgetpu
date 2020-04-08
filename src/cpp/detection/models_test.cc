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
      /*score_threshold=*/0.79f, /*iou_threshold=*/0.8f);
  TestCatMsCocoDetection(
      TestDataPath("ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite"),
      /*score_threshold=*/0.79f, /*iou_threshold=*/0.8f);

  // Mobilenet V2 SSD
  TestCatMsCocoDetection(
      TestDataPath("ssd_mobilenet_v2_coco_quant_postprocess.tflite"),
      /*score_threshold=*/0.95f, /*iou_threshold=*/0.86f);
  TestCatMsCocoDetection(
      TestDataPath("ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"),
      /*score_threshold=*/0.96f, /*iou_threshold=*/0.86f);
}

void TestFaceDetection(const std::string& model_name, float score_threshold,
                       float iou_threshold) {
  TestDetection(TestDataPath(model_name), TestDataPath("grace_hopper.bmp"),
                /*expected_box=*/{0.29f, 0.21f, 0.74f, 0.57f}, /*expected_label=*/0,
                score_threshold, iou_threshold);
}

TEST(DetectionEngineTest, TestFaceModel) {
  TestFaceDetection("ssd_mobilenet_v2_face_quant_postprocess.tflite", 0.7,
                    0.65f);
  TestFaceDetection("ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite",
                    0.7f, 0.65f);
}

TEST(DetectionEngineTest, TestFineTunedPetModel) {
  TestDetection(TestDataPath("ssd_mobilenet_v1_fine_tuned_pet.tflite"),
                TestDataPath("cat.bmp"),
                /*expected_box=*/{0.35f, 0.11f, 0.7f, 0.66f},
                /*expected_label=*/0, /*score_threshold=*/0.9,
                /*iou_threshold=*/0.81f);

  TestDetection(TestDataPath("ssd_mobilenet_v1_fine_tuned_pet_edgetpu.tflite"),
                TestDataPath("cat.bmp"),
                /*expected_box=*/{0.35f, 0.11f, 0.7f, 0.66f},
                /*expected_label=*/0, /*score_threshold=*/0.9f,
                /*iou_threshold=*/0.81f);
}

}  // namespace
}  // namespace coral

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
