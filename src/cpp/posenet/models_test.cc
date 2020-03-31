// Tests correctness of models.
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>

#include "absl/flags/parse.h"
#include "absl/strings/str_cat.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "src/cpp/basic/basic_engine.h"
#include "src/cpp/posenet/posenet_decoder_op.h"
#include "src/cpp/test_utils.h"

namespace coral {

struct Keypoint {
  Keypoint(float _y, float _x, float _score) : y(_y), x(_x), score(_score) {}
  float y;
  float x;
  float score;
};

void TestPoseNetDecoder(
    const std::string& model_name,
    const std::vector<std::array<Keypoint, 17>>& expected_poses,
    const float expected_pose_score) {
  const std::vector<std::vector<float>>& raw_outputs =
      TestWithImage(TestDataPath("posenet/" + model_name),
                    TestDataPath("posenet/test_image.bmp"));

  const auto& poses = raw_outputs[0];
  const auto& keypoint_scores = raw_outputs[1];
  const auto& pose_scores = raw_outputs[2];
  const auto& n_poses = raw_outputs[3];

  EXPECT_EQ(n_poses[0], 2);
  for (int i = 0; i < n_poses[0]; ++i) {
    EXPECT_GT(pose_scores[i], expected_pose_score);
    for (int k = 0; k < 17; k++) {
      if (expected_poses[i][k].score > 0.5) {
        // Expect scores to be close to the expected.
        EXPECT_NEAR(expected_poses[i][k].score, keypoint_scores[i * 17 + k],
                    0.05);
        // Compare positions only for points above a threshold score
        if (expected_poses[i][k].score > 0.1) {
          EXPECT_NEAR(expected_poses[i][k].y, poses[i * 17 * 2 + 2 * k], 3);
          EXPECT_NEAR(expected_poses[i][k].x, poses[i * 17 * 2 + 2 * k + 1],
                      3 /*within 3 pixels*/);
        }
      }
    }
  }
}

TEST(PosenetModelCorrectnessTest, TestPoseNetWithDecoder_353_481) {
  // test_image.bmp is royalty free from https://unsplash.com/photos/XuN44TajBGo
  // and shows two people standing. Expect keypoints to be roughly where
  // expected.
  std::vector<std::array<Keypoint, 17>> expected_poses(
      {std::array<Keypoint, 17>(
           {{Keypoint(97, 203, 0.984), Keypoint(93, 204, 0.751),
             Keypoint(93, 198, 0.980), Keypoint(96, 205, 0.033),
             Keypoint(95, 185, 0.986), Keypoint(122, 210, 0.995),
             Keypoint(127, 169, 0.996), Keypoint(161, 220, 0.978),
             Keypoint(165, 165, 0.994), Keypoint(194, 222, 0.976),
             Keypoint(201, 158, 0.990), Keypoint(198, 207, 0.994),
             Keypoint(199, 183, 0.991), Keypoint(247, 206, 0.914),
             Keypoint(248, 190, 0.944), Keypoint(287, 199, 0.077),
             Keypoint(308, 187, 0.044)}}),
       std::array<Keypoint, 17>(
           {{Keypoint(105, 257, 0.986), Keypoint(101, 262, 0.991),
             Keypoint(100, 256, 0.560), Keypoint(104, 277, 0.990),
             Keypoint(104, 256, 0.022), Keypoint(127, 289, 0.997),
             Keypoint(127, 255, 0.993), Keypoint(153, 300, 0.988),
             Keypoint(161, 245, 0.955), Keypoint(182, 309, 0.977),
             Keypoint(193, 233, 0.515), Keypoint(192, 285, 0.989),
             Keypoint(193, 258, 0.984), Keypoint(237, 282, 0.919),
             Keypoint(239, 270, 0.611), Keypoint(314, 286, 0.008),
             Keypoint(303, 288, 0.023)}})});
  TestPoseNetDecoder("posenet_mobilenet_v1_075_353_481_quant_decoder.tflite",
                     expected_poses, 0.70);
  TestPoseNetDecoder(
      "posenet_mobilenet_v1_075_353_481_quant_decoder_edgetpu.tflite",
      expected_poses, 0.70);
}

TEST(PosenetModelCorrectnessTest, TestPoseNetWithDecoder_481_641) {
  std::vector<std::array<Keypoint, 17>> expected_poses(
      {std::array<Keypoint, 17>(
           {{Keypoint(132, 271, 0.991), Keypoint(126, 273, 0.845),
             Keypoint(126, 264, 0.992), Keypoint(129, 275, 0.030),
             Keypoint(129, 245, 0.992), Keypoint(164, 281, 0.994),
             Keypoint(177, 225, 0.996), Keypoint(215, 294, 0.969),
             Keypoint(226, 217, 0.986), Keypoint(259, 298, 0.977),
             Keypoint(272, 209, 0.985), Keypoint(266, 281, 0.977),
             Keypoint(269, 246, 0.985), Keypoint(336, 277, 0.948),
             Keypoint(336, 255, 0.931), Keypoint(375, 271, 0.086),
             Keypoint(380, 258, 0.136)}}),
       std::array<Keypoint, 17>(
           {{Keypoint(143, 342, 0.996), Keypoint(136, 350, 0.996),
             Keypoint(137, 341, 0.827), Keypoint(139, 368, 0.995),
             Keypoint(141, 339, 0.008), Keypoint(171, 386, 0.996),
             Keypoint(173, 339, 0.995), Keypoint(211, 400, 0.991),
             Keypoint(215, 329, 0.876), Keypoint(247, 415, 0.948),
             Keypoint(259, 316, 0.182), Keypoint(256, 380, 0.989),
             Keypoint(257, 351, 0.989), Keypoint(344, 380, 0.016),
             Keypoint(341, 383, 0.012), Keypoint(426, 370, 0.002),
             Keypoint(416, 376, 0.003)}})});

  TestPoseNetDecoder("posenet_mobilenet_v1_075_481_641_quant_decoder.tflite",
                     expected_poses, 0.6);
  TestPoseNetDecoder(
      "posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite",
      expected_poses, 0.6);
}

TEST(PosenetModelCorrectnessTest, TestPoseNetWithDecoder_721_1281) {
  std::vector<std::array<Keypoint, 17>> expected_poses(
      {std::array<Keypoint, 17>(
           {{Keypoint(200, 543, 0.996), Keypoint(190, 543, 0.751),
             Keypoint(190, 530, 0.996), Keypoint(195, 546, 0.003),
             Keypoint(194, 495, 0.995), Keypoint(245, 568, 0.991),
             Keypoint(266, 450, 0.987), Keypoint(321, 592, 0.965),
             Keypoint(349, 427, 0.953), Keypoint(394, 608, 0.208),
             Keypoint(444, 410, 0.003), Keypoint(409, 564, 0.616),
             Keypoint(411, 486, 0.545), Keypoint(563, 630, 0.002),
             Keypoint(617, 421, 0.002), Keypoint(670, 651, 0.002),
             Keypoint(676, 425, 0.002)}}),
       std::array<Keypoint, 17>(
           {{Keypoint(211, 689, 0.992), Keypoint(199, 699, 0.994),
             Keypoint(204, 684, 0.876), Keypoint(205, 730, 0.987),
             Keypoint(217, 683, 0.029), Keypoint(259, 774, 0.908),
             Keypoint(268, 675, 0.927), Keypoint(315, 802, 0.912),
             Keypoint(332, 655, 0.643), Keypoint(373, 829, 0.346),
             Keypoint(420, 610, 0.008), Keypoint(393, 773, 0.222),
             Keypoint(391, 686, 0.201), Keypoint(475, 827, 0.002),
             Keypoint(582, 631, 0.002), Keypoint(632, 845, 0.002),
             Keypoint(674, 631, 0.002)}})});

  TestPoseNetDecoder("posenet_mobilenet_v1_075_721_1281_quant_decoder.tflite",
                     expected_poses, 0.45);
  TestPoseNetDecoder(
      "posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite",
      expected_poses, 0.45);
}

}  // namespace coral

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
