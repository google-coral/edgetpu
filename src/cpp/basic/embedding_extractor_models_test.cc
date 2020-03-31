// Tests correctness of embedding extractor models.

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
TEST(EmbeddingExtractorModelCorrectnessTest, TestMobilenetV1WithL2Norm) {
  BasicEngine engine(
      TestDataPath("mobilenet_v1_1.0_224_l2norm_quant_edgetpu.tflite"));
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
  EXPECT_GT(result[286], 0.66);  // Egyptian cat

  results = engine.RunInference(bird_input);
  ASSERT_EQ(1, results.size());
  result = results[0];
  class_max = std::distance(result.begin(),
                            std::max_element(result.begin(), result.end()));
  EXPECT_EQ(class_max, 20);
  EXPECT_GT(result[20], 0.9);  // chickadee
}

}  // namespace coral

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
