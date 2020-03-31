#include "src/cpp/basic/basic_engine.h"

#include <algorithm>
#include <chrono>  // NOLINT
#include <random>
#include <thread>  // NOLINT

#include "absl/flags/parse.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "src/cpp/fake_op.h"
#include "src/cpp/test_utils.h"

namespace coral {
namespace {

using ::testing::ElementsAre;

TEST(BasicEngineTest, TestDebugFunctions) {
  // Load the model.
  BasicEngine mobilenet_engine(
      TestDataPath("mobilenet_v1_1.0_224_quant.tflite"));
  // Check input dimensions.
  std::vector<int> input_tensor_shape =
      mobilenet_engine.get_input_tensor_shape();
  ASSERT_EQ(4, input_tensor_shape.size());
  EXPECT_THAT(input_tensor_shape, ElementsAre(1, 224, 224, 3));
  // Check output tensors.
  std::vector<size_t> output_tensor_sizes =
      mobilenet_engine.get_all_output_tensors_sizes();
  ASSERT_EQ(1, output_tensor_sizes.size());
  EXPECT_EQ(1001, output_tensor_sizes[0]);
  // Check model's path.
  EXPECT_EQ(TestDataPath("mobilenet_v1_1.0_224_quant.tflite"),
            mobilenet_engine.model_path());
}

TEST(BasicEngineTest, TestDebugFunctionsOnSsdModel) {
  // Test SSD model.
  BasicEngine ssd_mobilenet_engine(
      TestDataPath("ssd_mobilenet_v1_coco_quant_postprocess.tflite"));
  // Check input dimensions.
  std::vector<int> input_tensor_shape =
      ssd_mobilenet_engine.get_input_tensor_shape();
  ASSERT_EQ(4, input_tensor_shape.size());
  EXPECT_THAT(input_tensor_shape, ElementsAre(1, 300, 300, 3));
  std::vector<size_t> output_tensor_sizes =
      ssd_mobilenet_engine.get_all_output_tensors_sizes();
  // This SSD models is trained to recognize at most 20 bounding boxes.
  ASSERT_EQ(4, output_tensor_sizes.size());
  EXPECT_THAT(output_tensor_sizes, ElementsAre(80, 20, 20, 1));
}

TEST(BasicEngineTest, TwoEnginesSharedEdgeTpuSingleThreadInference) {
  // When there are multiple engines, their interpreters will share the Edge TPU
  // context. Ensure they can co-exist.
  BasicEngine engine_1(
      TestDataPath("mobilenet_v1_1.0_224_quant_edgetpu.tflite"));
  // `engine_2` shares the same Edge TPU with `engine_1`.
  BasicEngine engine_2(
      TestDataPath("mobilenet_v2_1.0_224_quant_edgetpu.tflite"),
      engine_1.device_path());
  // Tests with cat and bird.
  std::vector<uint8_t> cat_input =
      GetInputFromImage(TestDataPath("cat.bmp"), {224, 224, 3});
  std::vector<uint8_t> bird_input =
      GetInputFromImage(TestDataPath("bird.bmp"), {224, 224, 3});
  // Run inference alternately.
  std::vector<std::vector<float>> results;
  std::vector<float> result;
  for (int i = 0; i < 10; ++i) {
    results = engine_1.RunInference(cat_input);
    ASSERT_EQ(1, results.size());
    result = results[0];
    EXPECT_GT(result[286], 0.7);  // Egyptian cat

    results = engine_2.RunInference(cat_input);
    ASSERT_EQ(1, results.size());
    result = results[0];
    EXPECT_GT(result[286], 0.7);  // Egyptian cat

    results = engine_1.RunInference(bird_input);
    ASSERT_EQ(1, results.size());
    result = results[0];
    EXPECT_GT(result[20], 0.8);  // chickadee

    results = engine_2.RunInference(bird_input);
    ASSERT_EQ(1, results.size());
    result = results[0];
    EXPECT_GT(result[20], 0.8);  // chickadee
  }
}

// This test checks that when multiple interpreters in a multi-threaded
// environment, share the same Edge TPU. Each thread can receive correct result
// concurrently. There's no need to use Mutex to protect shared EdgeTpuContext
// because it is handled by underlying driver.
TEST(BasicEngineTest, TwoEnginesSharedEdgeTpuMultiThreadInference) {
  const auto &device_paths =
      EdgeTpuResourceManager::GetSingleton()->ListEdgeTpuPaths(
          EdgeTpuResourceManager::EdgeTpuState::kUnassigned);
  ASSERT_GE(device_paths.size(), 1);
  const auto &shared_device_path = device_paths[0];
  const int num_inferences = 1;

  // `job_a` runs iNat_bird model on a bird image. Sleep randomly between 2~20
  // ms after each inference.
  auto job_a =
      [&shared_device_path,
       num_inferences]() {  // NOLINT(clang-diagnostic-unused-lambda-capture)
        const auto &tid = std::this_thread::get_id();
        LOG(INFO) << "Thread: " << tid << " created.";
        BasicEngine engine(
            TestDataPath("mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite"),
            shared_device_path);
        const auto &input_shape = engine.get_input_tensor_shape();
        std::vector<uint8_t> bird_input =
            GetInputFromImage(TestDataPath("bird.bmp"),
                              {input_shape[1], input_shape[2], input_shape[3]});
        std::mt19937 generator(123456);
        std::uniform_int_distribution<> sleep_time_dist(2, 20);
        for (int i = 0; i < num_inferences; ++i) {
          const auto &results = engine.RunInference(bird_input);
          ASSERT_EQ(1, results.size());
          const auto &result = results[0];
          EXPECT_GT(result[659], 0.53);  // Black-capped Chickadee
          const auto sleep_time = sleep_time_dist(generator);
          std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
          LOG(INFO) << "Thread: " << tid << " sleep for " << sleep_time
                    << " ms.";
        }
        LOG(INFO) << "Thread: " << tid << " job done.";
      };

  // `job_b` runs iNat_insect model on a dragonfly image. Sleep randomly between
  // 1~10 ms. after each inference.
  auto job_b =
      [&shared_device_path,
       num_inferences]() {  // NOLINT(clang-diagnostic-unused-lambda-capture)
        const auto &tid = std::this_thread::get_id();
        LOG(INFO) << "Thread: " << tid << " created.";
        BasicEngine engine(
            TestDataPath(
                "mobilenet_v2_1.0_224_inat_insect_quant_edgetpu.tflite"),
            shared_device_path);
        const auto &input_shape = engine.get_input_tensor_shape();
        std::vector<uint8_t> dragonfly_input =
            GetInputFromImage(TestDataPath("dragonfly.bmp"),
                              {input_shape[1], input_shape[2], input_shape[3]});
        std::mt19937 generator(654321);
        std::uniform_int_distribution<> sleep_time_dist(1, 10);
        for (int i = 0; i < num_inferences; ++i) {
          const auto &results = engine.RunInference(dragonfly_input);
          ASSERT_EQ(1, results.size());
          const auto &result = results[0];
          EXPECT_GT(result[912], 0.25);  // Thornbush Dasher
          const auto sleep_time = sleep_time_dist(generator);
          std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
          LOG(INFO) << "Thread: " << tid << " sleep for " << sleep_time
                    << " ms.";
        }
        LOG(INFO) << "Thread: " << tid << " job done.";
      };

  std::thread thread_a(job_a);
  std::thread thread_b(job_b);
  std::thread thread_c(job_a);
  std::thread thread_d(job_b);

  thread_a.join();
  thread_b.join();
  thread_c.join();
  thread_d.join();
}

}  // namespace
}  // namespace coral

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
