// Tests correctness of float input models.

#include "absl/flags/parse.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "src/cpp/basic/basic_engine.h"
#include "src/cpp/test_utils.h"

namespace coral {
namespace {

TEST(ModelCorrectnessTest, MobilenetV1FloatInputs) {
  BasicEngine engine(
      TestDataPath("mobilenet_v1_1.0_224_ptq_float_io_edgetpu.tflite"));
  std::vector<uint8_t> cat_input =
      GetInputFromImage(TestDataPath("cat.bmp"), {224, 224, 3});
  std::vector<float> cat_input_float(cat_input.size());
  for (int i = 0; i < cat_input.size(); i++)
    cat_input_float.at(i) = (cat_input.at(i) - 127.5) / 127.5;
  std::vector<std::vector<float>> results =
      engine.RunInference(cat_input_float);
  ASSERT_EQ(1, results.size());
  std::vector<float> result = results[0];
  EXPECT_GT(result[286], 0.7);  // Egyptian cat

  std::vector<uint8_t> bird_input =
      GetInputFromImage(TestDataPath("bird.bmp"), {224, 224, 3});
  std::vector<float> bird_input_float(bird_input.size());
  for (int i = 0; i < bird_input.size(); i++)
    bird_input_float.at(i) = (bird_input.at(i) - 127.5) / 127.5;
  results = engine.RunInference(bird_input_float);
  ASSERT_EQ(1, results.size());
  result = results[0];
  EXPECT_GT(result[20], 0.9);  // chickadee
}

}  // namespace
}  // namespace coral

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
