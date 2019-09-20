// Stress test with multiple Edge TPU devices.
//
// By default, it launches one thread per Edge TPU devices it can find on the
// host system. And each thread will run `FLAGS_num_inferences` inferences.  It
// also checks the result returned is correct.

#include <functional>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "src/cpp/basic/basic_engine.h"
#include "src/cpp/test_utils.h"

ABSL_FLAG(int, num_inferences, 3000,
          "Number of inferences for each thread to run.");

namespace coral {
namespace {

using EdgeTpuState = coral::EdgeTpuResourceManager::EdgeTpuState;

typedef std::function<void(std::vector<std::vector<float>>)> CheckResultsFunc;

class MultipleTpusStressTest : public ::testing::Test {
 protected:
  void SetUp() override {
    resource_manager_ = coral::EdgeTpuResourceManager::GetSingleton();
    ASSERT_TRUE(resource_manager_);
    num_tpus_ =
        resource_manager_->ListEdgeTpuPaths(EdgeTpuState::kUnassigned).size();
    num_runs_ = absl::GetFlag(FLAGS_num_inferences);
    LOG(INFO) << "Each thread will run " << num_runs_ << " inferences.";
    workers_.reserve(num_tpus_);
  }

  void TearDown() override {
    EXPECT_EQ(
        0, resource_manager_->ListEdgeTpuPaths(EdgeTpuState::kAssigned).size());
  }

  void StressTest(const std::string& model_name,
                  const std::string& input_filename,
                  CheckResultsFunc check_results) {
    auto stress_test_job = [&]() {
      const auto& tid = std::this_thread::get_id();
      LOG(INFO) << "thread: " << tid << " created.";
      coral::BasicEngine engine(TestDataPath(model_name));
      const auto& input_shape = engine.get_input_tensor_shape();
      const auto& input_tensor = coral::GetInputFromImage(
          TestDataPath(input_filename),
          {input_shape[1], input_shape[2], input_shape[3]});
      std::vector<std::vector<float>> results;
      for (int i = 0; i < num_runs_; ++i) {
        results = engine.RunInference(input_tensor);
      }
      check_results(results);
      LOG(INFO) << "thread: " << tid << " done stress run.";
    };
    for (int i = 0; i < num_tpus_; ++i) {
      workers_.push_back(std::thread(stress_test_job));
    }
    for (int i = 0; i < num_tpus_; ++i) {
      workers_[i].join();
    }
    LOG(INFO) << "Stress test done for model: " << model_name;
  }

  coral::EdgeTpuResourceManager* resource_manager_ = nullptr;
  int num_tpus_ = 0;
  int num_runs_ = 0;
  std::vector<std::thread> workers_;
};

TEST_F(MultipleTpusStressTest, MobileNetV1) {
  CheckResultsFunc check_results =
      [](const std::vector<std::vector<float>>& results) {
        ASSERT_EQ(1, results.size());
        EXPECT_GT(results[0][286], 0.8);  // 286: Egyptian cat
      };
  StressTest("mobilenet_v1_1.0_224_quant_edgetpu.tflite", "cat.bmp",
             check_results);
}

TEST_F(MultipleTpusStressTest, MobileNetV2) {
  CheckResultsFunc check_results =
      [](const std::vector<std::vector<float>>& results) {
        ASSERT_EQ(1, results.size());
        EXPECT_GT(results[0][286], 0.79);  // 286: Egyptian cat
      };
  StressTest("mobilenet_v2_1.0_224_quant_edgetpu.tflite", "cat.bmp",
             check_results);
}

TEST_F(MultipleTpusStressTest, InceptionV1) {
  CheckResultsFunc check_results =
      [](const std::vector<std::vector<float>>& results) {
        ASSERT_EQ(1, results.size());
        EXPECT_GT(results[0][282], 0.38);  // 282: tabby, tabby cat
        EXPECT_GT(results[0][286], 0.38);  // 286: Egyptian cat
      };
  StressTest("inception_v1_224_quant_edgetpu.tflite", "cat.bmp", check_results);
}

TEST_F(MultipleTpusStressTest, InceptionV2) {
  CheckResultsFunc check_results =
      [](const std::vector<std::vector<float>>& results) {
        ASSERT_EQ(1, results.size());
        EXPECT_GT(results[0][282], 0.23);  // 282: tabby, tabby cat
        EXPECT_GT(results[0][286], 0.62);  // 286: Egyptian cat
      };
  StressTest("inception_v2_224_quant_edgetpu.tflite", "cat.bmp", check_results);
}

TEST_F(MultipleTpusStressTest, InceptionV3) {
  CheckResultsFunc check_results =
      [](const std::vector<std::vector<float>>& results) {
        ASSERT_EQ(1, results.size());
        EXPECT_GT(results[0][282], 0.59);  // 282: tabby, tabby cat
        EXPECT_GT(results[0][286], 0.29);  // 286: Egyptian cat
      };
  StressTest("inception_v3_299_quant_edgetpu.tflite", "cat.bmp", check_results);
}

TEST_F(MultipleTpusStressTest, InceptionV4) {
  CheckResultsFunc check_results =
      [](const std::vector<std::vector<float>>& results) {
        ASSERT_EQ(1, results.size());
        EXPECT_GT(results[0][282], 0.41);  // 282: tabby, tabby cat
        EXPECT_GT(results[0][286], 0.33);  // 286: Egyptian cat
      };
  StressTest("inception_v4_299_quant_edgetpu.tflite", "cat.bmp", check_results);
}

TEST_F(MultipleTpusStressTest, MobileNetV1Ssd) {
  CheckResultsFunc check_results =
      [](const std::vector<std::vector<float>>& results) {
        ASSERT_EQ(4, results.size());
        EXPECT_EQ(16, results[1][0]);  // 16: cat
        EXPECT_GT(results[2][0], 0.79);
      };
  StressTest("mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite",
             "cat.bmp", check_results);
}

TEST_F(MultipleTpusStressTest, MobileNetV2Ssd) {
  CheckResultsFunc check_results =
      [](const std::vector<std::vector<float>>& results) {
        ASSERT_EQ(4, results.size());
        EXPECT_EQ(16, results[1][0]);  // 16: cat
        EXPECT_GT(results[2][0], 0.96);
      };
  StressTest("mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite",
             "cat.bmp", check_results);
}

}  // namespace
}  // namespace coral

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
