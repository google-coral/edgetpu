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
#include "src/cpp/pipeline/pipelined_model_runner.h"
#include "src/cpp/pipeline/test_utils.h"
#include "src/cpp/pipeline/utils.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

ABSL_FLAG(std::string, model_names,
          "inception_v3_299_quant,"
          "inception_v4_299_quant",
          "Comma separated list of model names");

namespace coral {
namespace {

static constexpr char kPipelinedModelPrefix[] = "pipeline/";

#ifdef __arm__
static constexpr int kNumEdgeTpuAvailable = 2;
#else
static constexpr int kNumEdgeTpuAvailable = 4;
#endif

std::vector<int> NumSegments() {
  // `result` looks like 2, 3, ..., kNumEdgeTpuAvailable.
  std::vector<int> result(kNumEdgeTpuAvailable - 1);
  std::generate(result.begin(), result.end(),
                [n = 2]() mutable { return n++; });
  return result;
}

// Tests all supported models with different number of segments.
//
// The test parameter is number of segments a model is partitioned into.
class PipelinedModelRunnerModelsTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<int> {
 public:
  static void SetUpTestSuite() {
    input_tensors_map_ =
        new std::unordered_map<std::string, std::vector<PipelineTensor>>();

    ref_results_map_ =
        new std::unordered_map<std::string, std::vector<PipelineTensor>>();

    std::unordered_map<std::string, std::string> options = {
        {"Usb.MaxBulkInQueueLength", "8"},
    };
    const auto& available_tpus =
        edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
    CHECK_GE(available_tpus.size(), kNumEdgeTpuAvailable);
    edgetpu_resources_ =
        new std::vector<std::shared_ptr<edgetpu::EdgeTpuContext>>(
            kNumEdgeTpuAvailable);
    for (int i = 0; i < edgetpu_resources_->size(); ++i) {
      (*edgetpu_resources_)[i] =
          edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(
              available_tpus[i].type, available_tpus[i].path, options);
      LOG(INFO) << "Device " << available_tpus[i].path << " is selected.";
    }
  }

  static void TearDownTestSuite() {
    if (input_tensors_map_) {
      for (const auto& tensors : *input_tensors_map_) {
        for (const auto& tensor : tensors.second) {
          std::free(tensor.data.data);
        }
      }
      delete input_tensors_map_;
    }

    if (ref_results_map_) {
      for (const auto& tensors : *ref_results_map_) {
        for (const auto& tensor : tensors.second) {
          std::free(tensor.data.data);
        }
      }
      delete ref_results_map_;
    }

    if (edgetpu_resources_) {
      delete edgetpu_resources_;
    }
  }

  void SetUp() override {
    num_segments_ = GetParam();

    model_list_ = absl::StrSplit(absl::GetFlag(FLAGS_model_names), ',');
    for (const auto& model_base_name : model_list_) {
      // Calculate reference results.
      if (ref_results_map_->find(model_base_name) == ref_results_map_->end()) {
        // Construct tflite interpreter.
        const auto& model_name =
            absl::StrCat(model_base_name, "_edgetpu.tflite");
        auto model = tflite::FlatBufferModel::BuildFromFile(
            TestDataPath(model_name).c_str());
        auto interpreter =
            CreateInterpreter(*model, (*edgetpu_resources_)[0].get());

        // Setup input tensors.
        const auto& input_tensors = CreateRandomInputTensors(interpreter.get());
        input_tensors_map_->insert({model_base_name, input_tensors});
        for (int i = 0; i < interpreter->inputs().size(); ++i) {
          auto* tensor = interpreter->input_tensor(i);
          std::memcpy(tensor->data.data, input_tensors[i].data.data,
                      input_tensors[i].bytes);
        }

        CHECK(interpreter->Invoke() == kTfLiteOk);

        // Record reference results.
        std::vector<PipelineTensor> ref_results(interpreter->outputs().size());
        for (int i = 0; i < interpreter->outputs().size(); ++i) {
          auto* tensor = interpreter->output_tensor(i);
          ref_results[i].data.data = std::malloc(tensor->bytes);
          std::memcpy(ref_results[i].data.data, tensor->data.data,
                      tensor->bytes);
          ref_results[i].bytes = tensor->bytes;
          ref_results[i].type = tensor->type;
        }
        ref_results_map_->insert({model_base_name, ref_results});
      }
    }
  }

 protected:
  std::vector<PipelineTensor> RunInferenceWithPipelinedModel(
      const std::string& model_base_name) {
    CHECK_GE(edgetpu_resources_->size(), num_segments_);

    const auto& input_tensors = (*input_tensors_map_)[model_base_name];
    std::vector<std::unique_ptr<tflite::FlatBufferModel>> models(num_segments_);
    std::vector<std::unique_ptr<tflite::Interpreter>> managed_interpreters(
        num_segments_);
    std::vector<tflite::Interpreter*> interpreters(num_segments_);
    const auto& segments_names = SegmentsNames(model_base_name, num_segments_);

    // Construct PipelinedModelRunner.
    for (int i = 0; i < num_segments_; ++i) {
      models[i] = tflite::FlatBufferModel::BuildFromFile(
          TestDataPath(absl::StrCat(kPipelinedModelPrefix, segments_names[i]))
              .c_str());
      managed_interpreters[i] =
          CreateInterpreter(*(models[i]), (*edgetpu_resources_)[i].get());
      interpreters[i] = managed_interpreters[i].get();
    }
    runner_ = absl::make_unique<PipelinedModelRunner>(interpreters);

    // Run inference.
    runner_->Push(CopyTensors(input_tensors));
    std::vector<PipelineTensor> output_tensors;
    runner_->Pop(&output_tensors);
    return output_tensors;
  }

  void CheckSameTensors(const std::vector<PipelineTensor>& actual_tensors,
                        const std::vector<PipelineTensor>& expected_tensors) {
    ASSERT_EQ(actual_tensors.size(), expected_tensors.size());
    for (int i = 0; i < expected_tensors.size(); ++i) {
      EXPECT_EQ(actual_tensors[i].type, expected_tensors[i].type);
      ASSERT_EQ(actual_tensors[i].bytes, expected_tensors[i].bytes);
      const auto* actual =
          reinterpret_cast<const uint8_t*>(actual_tensors[i].data.data);
      const auto* expected =
          reinterpret_cast<const uint8_t*>(expected_tensors[i].data.data);
      for (int j = 0; j < expected_tensors[i].bytes; ++j) {
        EXPECT_EQ(actual[j], expected[j]);
      }
    }
  }

  std::vector<PipelineTensor> CopyTensors(
      const std::vector<PipelineTensor>& tensors) {
    std::vector<PipelineTensor> copy(tensors.size());
    for (int i = 0; i < tensors.size(); ++i) {
      copy[i].data.data =
          runner_->GetInputTensorAllocator()->alloc(tensors[i].bytes);
      copy[i].bytes = tensors[i].bytes;
      copy[i].type = tensors[i].type;
      std::memcpy(copy[i].data.data, tensors[i].data.data, tensors[i].bytes);
    }
    return copy;
  }

  // List of models to test.
  std::vector<std::string> model_list_;

  // Key is `model_list_[i]`, value is input tensors. This map makes sure that
  // the same input tensors are used when running the model with no partition, 2
  // partitins, 3 partitions, and so on.
  static std::unordered_map<std::string, std::vector<PipelineTensor>>*
      input_tensors_map_;

  // Key is `model_list_[i]`, value is output tensors from non partitioned
  // model, recorded here as reference results.
  static std::unordered_map<std::string, std::vector<PipelineTensor>>*
      ref_results_map_;

  // Cache Edge TPUs to improve test performance.
  static std::vector<std::shared_ptr<edgetpu::EdgeTpuContext>>*
      edgetpu_resources_;

  int num_segments_;

  std::unique_ptr<PipelinedModelRunner> runner_;
};

std::unordered_map<std::string, std::vector<PipelineTensor>>*
    PipelinedModelRunnerModelsTest::input_tensors_map_ = nullptr;
std::unordered_map<std::string, std::vector<PipelineTensor>>*
    PipelinedModelRunnerModelsTest::ref_results_map_ = nullptr;
std::vector<std::shared_ptr<edgetpu::EdgeTpuContext>>*
    PipelinedModelRunnerModelsTest::edgetpu_resources_ = nullptr;

TEST_P(PipelinedModelRunnerModelsTest, CheckInferenceResult) {
  for (const auto& model_base_name : model_list_) {
    LOG(INFO) << "Testing " << model_base_name << " with " << num_segments_
              << " segments.";
    const auto& output_tensors =
        RunInferenceWithPipelinedModel(model_base_name);
    const auto& expected_tensors = (*ref_results_map_)[model_base_name];
    CheckSameTensors(output_tensors, expected_tensors);
    FreeTensors(output_tensors, runner_->GetOutputTensorAllocator());
  }
}

TEST_P(PipelinedModelRunnerModelsTest, RepeatabilityTest) {
  constexpr int kNumRuns = 10;
  for (const auto& model_base_name : model_list_) {
    LOG(INFO) << "Testing " << model_base_name << " with " << num_segments_
              << " segments.";
    const auto& expected_tensors =
        RunInferenceWithPipelinedModel(model_base_name);
    for (int i = 0; i < kNumRuns; ++i) {
      const auto& output_tensors =
          RunInferenceWithPipelinedModel(model_base_name);
      CheckSameTensors(output_tensors, expected_tensors);
      FreeTensors(output_tensors, runner_->GetOutputTensorAllocator());
    }
    FreeTensors(expected_tensors, runner_->GetOutputTensorAllocator());
  }
}

INSTANTIATE_TEST_CASE_P(PipelinedModelRunnerModelsTest,
                        PipelinedModelRunnerModelsTest,
                        ::testing::ValuesIn(NumSegments()));

}  // namespace
}  // namespace coral

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
