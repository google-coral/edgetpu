#include "src/cpp/pipeline/pipelined_model_runner.h"

#include <memory>
#include <random>
#include <thread>  // NOLINT
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "edgetpu.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "src/cpp/pipeline/common.h"
#include "src/cpp/pipeline/test_utils.h"
#include "src/cpp/pipeline/utils.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

namespace coral {
namespace {

static constexpr char kPipelinedModelPrefix[] = "pipeline/";

#ifdef __arm__
static constexpr int kNumEdgeTpuAvailable = 2;
#else
static constexpr int kNumEdgeTpuAvailable = 4;
#endif

// Tests to make sure `PipelinedModelRunner` API works as stated.
class PipelinedModelRunnerApiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const int num_segments = kNumEdgeTpuAvailable;
    const std::string model_name = "inception_v4_299_quant";

    // Grab Edge TPU contexts.
    std::unordered_map<std::string, std::string> options = {
        {"Usb.MaxBulkInQueueLength", "8"},
    };
    const auto& available_tpus =
        edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
    CHECK_GE(available_tpus.size(), num_segments);
    edgetpu_resources_.resize(num_segments);
    for (int i = 0; i < num_segments; ++i) {
      edgetpu_resources_[i] =
          edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(
              available_tpus[i].type, available_tpus[i].path, options);
      LOG(INFO) << "Device " << available_tpus[i].path << " is selected.";
    }

    // Construct PipelinedModelRunner.
    const auto& segments_names = SegmentsNames(model_name, num_segments);
    std::vector<tflite::Interpreter*> interpreters(num_segments);
    models_.resize(num_segments);
    managed_interpreters_.resize(num_segments);
    for (int i = 0; i < num_segments; ++i) {
      models_[i] = tflite::FlatBufferModel::BuildFromFile(
          TestDataPath(absl::StrCat(kPipelinedModelPrefix, segments_names[i]))
              .c_str());
      managed_interpreters_[i] =
          CreateInterpreter(*(models_[i]), edgetpu_resources_[i].get());
      interpreters[i] = managed_interpreters_[i].get();
    }
    runner_ = absl::make_unique<PipelinedModelRunner>(interpreters);
  }

  std::vector<std::shared_ptr<edgetpu::EdgeTpuContext>> edgetpu_resources_;
  std::vector<std::unique_ptr<tflite::FlatBufferModel>> models_;
  std::vector<std::unique_ptr<tflite::Interpreter>> managed_interpreters_;
  std::unique_ptr<PipelinedModelRunner> runner_;
};

TEST_F(PipelinedModelRunnerApiTest, PushValidRequest) {
  const auto& input_tensors = CreateRandomInputTensors(
      managed_interpreters_[0].get(), runner_->GetInputTensorAllocator());
  EXPECT_TRUE(runner_->Push(input_tensors));
  std::vector<PipelineTensor> output_tensors;
  ASSERT_TRUE(runner_->Pop(&output_tensors));
  EXPECT_EQ(output_tensors.size(), 1);

  const auto& stats_list = runner_->GetSegmentStats();
  EXPECT_EQ(stats_list.size(), managed_interpreters_.size());
  for (const auto& stats : stats_list) {
    EXPECT_EQ(stats.num_inferences, 1);
    EXPECT_GT(stats.total_time_ns, 0);
  }
}

TEST_F(PipelinedModelRunnerApiTest, PushOneEmptyRequest) {
  EXPECT_TRUE(runner_->Push({}));
  // Pushing after empty request should return false.
  EXPECT_FALSE(runner_->Push({}));
  std::vector<PipelineTensor> output_tensors;
  ASSERT_FALSE(runner_->Pop(&output_tensors));
  EXPECT_EQ(output_tensors.size(), 0);
  FreeTensors(output_tensors, runner_->GetOutputTensorAllocator());

  const auto& stats_list = runner_->GetSegmentStats();
  EXPECT_EQ(stats_list.size(), managed_interpreters_.size());
  for (const auto& stats : stats_list) {
    EXPECT_EQ(stats.num_inferences, 0);
    EXPECT_EQ(stats.total_time_ns, 0);
  }
}

TEST_F(PipelinedModelRunnerApiTest, PushTwoEmptyRequests) {
  auto thread_task = [this](bool* push_status) {
    *push_status = runner_->Push({});
    LOG(INFO) << "Thread: " << std::this_thread::get_id()
              << " returned push status:  " << *push_status;
  };

  bool push_status[2] = {true, true};
  auto thread_0 = std::thread(thread_task, &push_status[0]);
  auto thread_1 = std::thread(thread_task, &push_status[1]);
  thread_0.join();
  thread_1.join();

  // Only one push status should be true.
  EXPECT_NE(push_status[0], push_status[1]);

  const auto& stats_list = runner_->GetSegmentStats();
  EXPECT_EQ(stats_list.size(), managed_interpreters_.size());
  for (const auto& stats : stats_list) {
    EXPECT_EQ(stats.num_inferences, 0);
    EXPECT_EQ(stats.total_time_ns, 0);
  }
}

TEST_F(PipelinedModelRunnerApiTest, PushEmptyAndValidRequests) {
  auto push_empty_request = [this](bool* push_status) {
    LOG(INFO) << "Thread: " << std::this_thread::get_id()
              << " pushing empty request...";
    *push_status = runner_->Push({});
    LOG(INFO) << "Thread: " << std::this_thread::get_id()
              << " returned push status:  " << *push_status;
  };

  auto push_valid_request = [this](bool* push_status) {
    LOG(INFO) << "Thread: " << std::this_thread::get_id()
              << " pushing valid request...";
    const auto& input_tensors = CreateRandomInputTensors(
        managed_interpreters_[0].get(), runner_->GetInputTensorAllocator());
    *push_status = runner_->Push(input_tensors);
    LOG(INFO) << "Thread: " << std::this_thread::get_id()
              << " returned push status:  " << *push_status;
    // Free input_tensors if it was not pushed successfully.
    if (!(*push_status)) {
      FreeTensors(input_tensors, runner_->GetInputTensorAllocator());
    }
  };

  bool push_status[2] = {true, true};
  auto thread_0 = std::thread(push_empty_request, &push_status[0]);
  auto thread_1 = std::thread(push_valid_request, &push_status[1]);
  thread_0.join();
  thread_1.join();

  // Empty request should always returns true. Valid request may return true or
  // false depending on whether it pushes before empty request is processed.
  EXPECT_TRUE(push_status[0]);
}

}  // namespace
}  // namespace coral

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
