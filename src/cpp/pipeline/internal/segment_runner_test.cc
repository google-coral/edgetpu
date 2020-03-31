#include "src/cpp/pipeline/internal/segment_runner.h"

#include <memory>

#include "absl/flags/parse.h"
#include "absl/memory/memory.h"
#include "edgetpu.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "src/cpp/pipeline/internal/default_allocator.h"
#include "src/cpp/pipeline/test_utils.h"
#include "src/cpp/pipeline/utils.h"
#include "tensorflow/lite/interpreter.h"

namespace coral {
namespace internal {
namespace {

TensorMap GenerateRandomInputs(
    const tflite::Interpreter& interpreter,
    const std::unordered_map<std::string, int>& tensor_consumers_count,
    Allocator* allocator) {
  TensorMap input_tensors;
  for (int input_index : interpreter.inputs()) {
    const auto* tflite_tensor = interpreter.tensor(input_index);
    PipelineTensor input_tensor;
    input_tensor.data.data =
        CHECK_NOTNULL(allocator->alloc(tflite_tensor->bytes));
    input_tensor.bytes = tflite_tensor->bytes;
    input_tensor.type = tflite_tensor->type;
    FillRandom(reinterpret_cast<uint8_t*>(input_tensor.data.data),
               input_tensor.bytes);
    auto it = tensor_consumers_count.find(tflite_tensor->name);
    CHECK(it != tensor_consumers_count.end());
    input_tensors.insert(
        {tflite_tensor->name, {input_tensor, /*num_consumers=*/it->second}});
  }
  return input_tensors;
}

std::unordered_map<std::string, int> BuildTensorConsumersCountMap(
    const std::unordered_set<std::string>& input_tensor_names) {
  std::unordered_map<std::string, int> tensor_consumers_count(
      input_tensor_names.size());
  for (const auto& tensor_name : input_tensor_names) {
    tensor_consumers_count[tensor_name] = 1;
  }
  return tensor_consumers_count;
}

void FreeTensors(const TensorMap& tensors, Allocator* allocator) {
  for (const auto& pair : tensors) {
    const auto& tensor = pair.second;
    allocator->free(tensor.tensor.data.data, tensor.tensor.bytes);
  }
}

void CheckResults(const tflite::Interpreter& interpreter,
                  const TensorMap& tensors) {
  ASSERT_LE(interpreter.outputs().size(), tensors.size());
  for (int i = 0; i < interpreter.outputs().size(); ++i) {
    const auto* tflite_tensor = interpreter.output_tensor(i);
    const auto& it = tensors.find(tflite_tensor->name);
    ASSERT_NE(it, tensors.end());
    ASSERT_EQ(it->second.tensor.bytes, tflite_tensor->bytes);
    const auto* actual =
        reinterpret_cast<const uint8_t*>(it->second.tensor.data.data);
    const auto* expected =
        reinterpret_cast<const uint8_t*>(tflite_tensor->data.data);
    for (int j = 0; j < tflite_tensor->bytes; ++j) {
      EXPECT_EQ(actual[j], expected[j]);
    }
  }
}

// Tests that SegmentRunner returns same output tensors as using
// tflite::Interpreter when feeding the same inputs.
TEST(SegmentRunnerTest, SameResultAsTfliteInterpreter) {
  const std::string model_name = "inception_v4_299_quant_edgetpu.tflite";
  auto model =
      tflite::FlatBufferModel::BuildFromFile(TestDataPath(model_name).c_str());
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_resource =
      edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();

  auto pipeline_interpreter = CreateInterpreter(*model, edgetpu_resource.get());
  auto tflite_interpreter = CreateInterpreter(*model, edgetpu_resource.get());

  // Create SegmentRunner.
  WaitQueue<TensorMap> input_queue, output_queue;
  const auto& input_tensor_names = GetInputTensorNames(*pipeline_interpreter);
  auto tensor_consumers_count =
      BuildTensorConsumersCountMap(input_tensor_names);
  // Add a fake tensor to make a harder case.
  tensor_consumers_count.insert({"fake_tensor_name", 20});
  auto default_allocator = absl::make_unique<DefaultAllocator>();
  SegmentRunner runner = {
      &tensor_consumers_count,
      &input_tensor_names,
      pipeline_interpreter.get(),
      &input_queue,
      &output_queue,
      default_allocator.get(),
      default_allocator.get(),
  };

  // Set up input for SegmentRunner.
  const auto& input_tensors = GenerateRandomInputs(
      *pipeline_interpreter, tensor_consumers_count, default_allocator.get());
  input_queue.push(input_tensors);
  // Promise that no more requests will be added.
  input_queue.StopWaiters();

  // Let `tflite_interpreter` uses the same random inputs.
  for (int i = 0; i < pipeline_interpreter->inputs().size(); ++i) {
    auto* tflite_tensor = tflite_interpreter->input_tensor(i);
    auto it = input_tensors.find(tflite_tensor->name);
    CHECK(it != input_tensors.end());
    std::memcpy(tflite_tensor->data.data, it->second.tensor.data.data,
                it->second.tensor.bytes);
  }

  // Run inference with SegmentRunner.
  runner.RunInference();
  EXPECT_EQ(runner.stats().num_inferences, 1);
  EXPECT_GT(runner.stats().total_time_ns, 0);

  // Run inference with tflite::Interpreter.
  CHECK(tflite_interpreter->Invoke() == kTfLiteOk);

  // Check that results are exactly the same.
  ASSERT_EQ(output_queue.size(), 1);
  TensorMap output_tensors;
  ASSERT_TRUE(output_queue.Wait(&output_tensors));
  CheckResults(*tflite_interpreter, output_tensors);
  FreeTensors(output_tensors, default_allocator.get());
}

class MockAllocator : public Allocator {
 public:
  MOCK_METHOD(void*, alloc, (size_t), (override));
  MOCK_METHOD(void, free, (void*, size_t), (override));
};

// Class that issues expected (valid) memory address for MockAllocator based on
// tflite::Interpreter.
class AddressCalculator {
 public:
  explicit AddressCalculator(const tflite::Interpreter* interpreter)
      : interpreter_(interpreter) {}

  ~AddressCalculator() {
    for (auto* addr : allocated_memory_list_) {
      std::free(addr);
    }
  }

  // Allocates buffer for input tensor whose index is `i`.
  void* alloc_input(int i) {
    CHECK_LT(i, interpreter_->inputs().size());
    auto* addr = std::malloc(interpreter_->input_tensor(i)->bytes);
    allocated_memory_list_.push_back(addr);
    return addr;
  }

  // Returns size (in bytes) for input tensor whose index is `i`.
  size_t input_size(int i) { return interpreter_->input_tensor(i)->bytes; }

  // Allocates buffer for output tensor whose index is `i`.
  void* alloc_output(int i) {
    CHECK_LT(i, interpreter_->outputs().size());
    auto* addr = std::malloc(interpreter_->output_tensor(i)->bytes);
    allocated_memory_list_.push_back(addr);
    return addr;
  }

  // Returns size (in bytes) for output tensor whose index is `i`.
  size_t output_size(int i) { return interpreter_->output_tensor(i)->bytes; }

 private:
  const tflite::Interpreter* interpreter_ = nullptr;
  std::vector<void*> allocated_memory_list_;
};

TEST(SegmentRunnerTest, InputTensorsFreedByRunner) {
  const std::string model_name = "inception_v4_299_quant_edgetpu.tflite";
  auto model =
      tflite::FlatBufferModel::BuildFromFile(TestDataPath(model_name).c_str());
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_resource =
      edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
  auto interpreter = CreateInterpreter(*model, edgetpu_resource.get());

  // Create SegmentRunner.
  WaitQueue<TensorMap> input_queue, output_queue;
  const auto& input_tensor_names = GetInputTensorNames(*interpreter);
  const auto& tensor_consumers_count =
      BuildTensorConsumersCountMap(input_tensor_names);
  auto mock_allocator = absl::make_unique<MockAllocator>();
  SegmentRunner runner = {
      &tensor_consumers_count, &input_tensor_names,
      interpreter.get(),       &input_queue,
      &output_queue,           mock_allocator.get(),
      mock_allocator.get(),
  };

  // Set expectation for `mock_allocator`
  auto addr_calculator = AddressCalculator(interpreter.get());
  auto* input_tensor_addr = addr_calculator.alloc_input(0);
  size_t input_tensor_size = addr_calculator.input_size(0);
  auto* output_tensor_addr = addr_calculator.alloc_output(0);
  EXPECT_CALL(*mock_allocator, alloc)
      .Times(2)
      .WillOnce(testing::Return(input_tensor_addr))
      .WillOnce(testing::Return(output_tensor_addr));
  EXPECT_CALL(*mock_allocator, free(input_tensor_addr, input_tensor_size))
      .Times(1);

  // Set up input for SegmentRunner.
  const auto& input_tensors = GenerateRandomInputs(
      *interpreter, tensor_consumers_count, mock_allocator.get());
  input_queue.push(input_tensors);
  // Promise that no more requests will be added.
  input_queue.StopWaiters();

  // Run inference with SegmentRunner.
  runner.RunInference();
  EXPECT_EQ(runner.stats().num_inferences, 1);
  EXPECT_GT(runner.stats().total_time_ns, 0);

  // Check output tensors.
  ASSERT_EQ(output_queue.size(), 1);
  TensorMap output_tensors;
  ASSERT_TRUE(output_queue.Wait(&output_tensors));
  ASSERT_EQ(output_tensors.size(), 1);
  const auto& output_tensor = output_tensors.begin()->second;
  EXPECT_EQ(output_tensor.num_consumers, 0);
  EXPECT_EQ(output_tensor.tensor.data.data, output_tensor_addr);

  // Ideally, one should free `output_tensors` after consuming them, e.g.,
  // `FreeTensors(output_tensors, mock_allocator.get()). This step is skipped
  // here on purpose to make sure only `input_tensor_addr` was freed insided
  // SegmentRunner::RunInference().
}

// Set input tensors' num_consumer count >1 on purpose to see it is kept alive
// after call to RunInference().
TEST(SegmentRunnerTest, InputTensorsKeptAliveByRunner) {
  const std::string model_name = "inception_v4_299_quant_edgetpu.tflite";
  auto model =
      tflite::FlatBufferModel::BuildFromFile(TestDataPath(model_name).c_str());
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_resource =
      edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
  auto interpreter = CreateInterpreter(*model, edgetpu_resource.get());

  // Create SegmentRunner.
  WaitQueue<TensorMap> input_queue, output_queue;
  const auto& input_tensor_names = GetInputTensorNames(*interpreter);
  const auto& tensor_consumers_count =
      BuildTensorConsumersCountMap(input_tensor_names);
  auto mock_allocator = absl::make_unique<MockAllocator>();
  SegmentRunner runner = {
      &tensor_consumers_count, &input_tensor_names,
      interpreter.get(),       &input_queue,
      &output_queue,           mock_allocator.get(),
      mock_allocator.get(),
  };

  // Set expectation for `mock_allocator`
  auto addr_calculator = AddressCalculator(interpreter.get());
  auto* input_tensor_addr = addr_calculator.alloc_input(0);
  auto* output_tensor_addr = addr_calculator.alloc_output(0);
  EXPECT_CALL(*mock_allocator, alloc)
      .Times(2)
      .WillOnce(testing::Return(input_tensor_addr))
      .WillOnce(testing::Return(output_tensor_addr));
  EXPECT_CALL(*mock_allocator, free(testing::Ne(nullptr), testing::Gt(0)))
      .Times(0);

  // Set up input for SegmentRunner.
  auto input_tensors = GenerateRandomInputs(
      *interpreter, tensor_consumers_count, mock_allocator.get());
  for (auto& pair : input_tensors) {
    pair.second.num_consumers = 2;
  }
  input_queue.push(input_tensors);
  // Promise that no more requests will be added.
  input_queue.StopWaiters();

  // Run inference with SegmentRunner.
  runner.RunInference();
  EXPECT_EQ(runner.stats().num_inferences, 1);
  EXPECT_GT(runner.stats().total_time_ns, 0);

  // Check output tensors.
  ASSERT_EQ(output_queue.size(), 1);
  TensorMap output_tensors;
  ASSERT_TRUE(output_queue.Wait(&output_tensors));
  ASSERT_EQ(output_tensors.size(), 2);

  ASSERT_EQ(input_tensors.size(), 1);
  const auto& unconsumed_input_tensor =
      output_tensors.at(interpreter->input_tensor(0)->name);
  EXPECT_EQ(unconsumed_input_tensor.num_consumers, 1);
  EXPECT_EQ(unconsumed_input_tensor.tensor.data.data, input_tensor_addr);

  const auto& real_output_tensor =
      output_tensors.at(interpreter->output_tensor(0)->name);
  EXPECT_EQ(real_output_tensor.num_consumers, 0);
  EXPECT_EQ(real_output_tensor.tensor.data.data, output_tensor_addr);

  // Ideally, one should free `input_tensors` and `output_tensors` after
  // consuming them, e.g., `FreeTensors(output_tensors, mock_allocator.get()).
  // This step is skipped here on purpose to make sure Allocator::free() was not
  // called insided SegmentRunner::RunInference().
}

}  // namespace
}  // namespace internal
}  // namespace coral

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
