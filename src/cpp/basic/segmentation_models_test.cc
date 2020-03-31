// Tests correctness of image segmentation models.

#include <cmath>
#include <iostream>

#include "absl/flags/parse.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "src/cpp/basic/basic_engine.h"
#include "src/cpp/test_utils.h"

namespace coral {

TEST(ModelCorrectnessTest, Deeplab513Mv2Dm1_WithArgMax) {
  // See label map:
  // test_data/pascal_voc_segmentation_labels.txt
  std::vector<uint8_t> cpu_pred_segmentation;
  TestSegmentation("deeplabv3_mnv2_pascal_quant.tflite",
                   "bird_segmentation.bmp", "bird_segmentation_mask.bmp",
                   /*size=*/513,
                   /*iou_threshold=*/0.9,
                   /*model_has_argmax=*/true, &cpu_pred_segmentation);
  std::vector<uint8_t> edgetpu_pred_segmentation;
  TestSegmentation("deeplabv3_mnv2_pascal_quant_edgetpu.tflite",
                   "bird_segmentation.bmp", "bird_segmentation_mask.bmp",
                   /*size=*/513,
                   /*iou_threshold=*/0.9,
                   /*model_has_argmax=*/true, &edgetpu_pred_segmentation);
  float seg_iou = ComputeIntersectionOverUnion(cpu_pred_segmentation,
                                               edgetpu_pred_segmentation);
  EXPECT_GT(seg_iou, 0.99);
}

TEST(ModelCorrectnessTest, Deeplab513Mv2Dm05_WithArgMax) {
  // See label map:
  // test_data/pascal_voc_segmentation_labels.txt
  std::vector<uint8_t> cpu_pred_segmentation;
  TestSegmentation("deeplabv3_mnv2_dm05_pascal_quant.tflite",
                   "bird_segmentation.bmp", "bird_segmentation_mask.bmp",
                   /*size=*/513,
                   /*iou_threshold=*/0.9,
                   /*model_has_argmax=*/true, &cpu_pred_segmentation);
  std::vector<uint8_t> edgetpu_pred_segmentation;
  TestSegmentation("deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite",
                   "bird_segmentation.bmp", "bird_segmentation_mask.bmp",
                   /*size=*/513,
                   /*iou_threshold=*/0.9,
                   /*model_has_argmax=*/true, &edgetpu_pred_segmentation);
  float seg_iou = ComputeIntersectionOverUnion(cpu_pred_segmentation,
                                               edgetpu_pred_segmentation);
  EXPECT_GT(seg_iou, 0.98);
}

// Tests the corretness of an example U-Net model trained following the tutorial
// on https://www.tensorflow.org/tutorials/images/segmentation.
TEST(ModelCorrectnessTest, Keras_PostTrainingQuantization_UNet128MobilenetV2) {
  // The masks are basically labels for each pixel. Each pixel is given one of
  // three categories :
  // Class 1 : Pixel belonging to the pet.
  // Class 2 : Pixel bordering the pet.
  // Class 3 : None of the above/ Surrounding pixel.
  std::vector<uint8_t> cpu_pred_segmentation;
  TestSegmentation("keras_post_training_unet_mv2_128_quant.tflite",
                   "dog_segmentation.bmp", "dog_segmentation_mask.bmp",
                   /*size=*/128,
                   /*iou_threshold=*/0.86,
                   /*model_has_argmax=*/false, &cpu_pred_segmentation);
  std::vector<uint8_t> edgetpu_pred_segmentation;
  TestSegmentation("keras_post_training_unet_mv2_128_quant_edgetpu.tflite",
                   "dog_segmentation.bmp", "dog_segmentation_mask.bmp",
                   /*size=*/128,
                   /*iou_threshold=*/0.86,
                   /*model_has_argmax=*/false, &edgetpu_pred_segmentation);
  float seg_iou = ComputeIntersectionOverUnion(cpu_pred_segmentation,
                                               edgetpu_pred_segmentation);
  EXPECT_GT(seg_iou, 0.97);
}

}  // namespace coral

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
