#include "src/cpp/pipeline/pipelined_model_runner.h"

#include <thread>  // NOLINT
#include <unordered_map>
#include <unordered_set>

#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "glog/logging.h"
#include "src/cpp/pipeline/internal/default_allocator.h"
#include "src/cpp/pipeline/internal/memory_pool_allocator.h"
#include "src/cpp/pipeline/internal/segment_runner.h"
#include "src/cpp/pipeline/utils.h"
#include "tensorflow/lite/interpreter.h"

namespace coral {

using coral::internal::TensorMap;

PipelinedModelRunner::PipelinedModelRunner(
    const std::vector<tflite::Interpreter*>& model_segments_interpreters,
    Allocator* input_tensor_allocator, Allocator* output_tensor_allocator)
    : segments_interpreters_(model_segments_interpreters),
      num_segments_(model_segments_interpreters.size()),
      queues_(num_segments_ + 1),
      threads_(num_segments_),
      segments_runners_(num_segments_),
      input_tensor_names_per_segment_(num_segments_),
      input_tensor_allocator_(input_tensor_allocator),
      output_tensor_allocator_(output_tensor_allocator) {
  if ((input_tensor_allocator_ == nullptr) ||
      (output_tensor_allocator_ == nullptr)) {
    default_allocator_ = absl::make_unique<internal::DefaultAllocator>();
  }
  if (input_tensor_allocator_ == nullptr) {
    input_tensor_allocator_ = default_allocator_.get();
  }
  if (output_tensor_allocator_ == nullptr) {
    output_tensor_allocator_ = default_allocator_.get();
  }

  // Size of runner's input and output queue must be unbounded as callers are
  // allowed to push faster than processing speed of pipeline, and caller are
  // allowed to consume the result in any pace they choose. However,
  // intermediate tensor queues can be set to be bounded to avoid possible
  // high memory cost due to unbalanced model partitions. And a queue size of 1
  // is enough to make sure pipeline can function efficiently.
  const int kInternalQueueSize = 1;
  VLOG(1) << "Setting internal queue size at: " << kInternalQueueSize;
  for (int i = 1; i < num_segments_; ++i) {
    queues_[i].set_max_queue_size(kInternalQueueSize);
  }

  VLOG(1) << "Finding last segments that consumes a tensor...";
  // Key is (output) tensor name, value is the last segment that consumes it.
  std::unordered_map<std::string, int> last_consumed_by_map;
  for (int i = 0; i < num_segments_ - 1; ++i) {
    for (const int output_index : segments_interpreters_[i]->outputs()) {
      const auto* output_name =
          segments_interpreters_[i]->tensor(output_index)->name;

      // Search for the last segment that consumes `output_name`.
      for (int j = i + 1; j < num_segments_; ++j) {
        if (GetInputTensor(*(segments_interpreters_[j]), output_name)) {
          last_consumed_by_map[output_name] = j;
          break;
        }
      }

      // Sanity check.
      CHECK(last_consumed_by_map.find(output_name) !=
            last_consumed_by_map.end());
      CHECK_GT(last_consumed_by_map[output_name], i)
          << "Output tensor " << output_name
          << " must be consumed by subsequent segment.";
    }
  }

  VLOG(1) << "Calculating intermediate tensors buffer size...";
  std::unordered_map<size_t, int> tensor_size_to_copy_map;
  for (int i = 0; i < num_segments_ - 1; ++i) {
    VLOG(1) << "Analyzing output tensors of segment " << i;
    for (const int index : segments_interpreters_[i]->outputs()) {
      const auto* tensor = segments_interpreters_[i]->tensor(index);
      int last_consumed_by = last_consumed_by_map.at(tensor->name);
      int copies = 1 + (last_consumed_by - i) * (1 + kInternalQueueSize);
      tensor_size_to_copy_map[tensor->bytes] += copies;
      VLOG(1) << "tensor name: " << tensor->name
              << " size (bytes): " << tensor->bytes << " copies: " << copies;
    }
  }
  intermediate_tensor_allocator_ =
      absl::make_unique<internal::MemoryPoolAllocator>(tensor_size_to_copy_map);

  VLOG(1) << "Analyzing consumers for all input/intermediate tensors...";
  for (int i = 0; i < num_segments_; ++i) {
    CHECK(segments_interpreters_[i]);
    input_tensor_names_per_segment_[i] =
        GetInputTensorNames(*segments_interpreters_[i]);
    for (const auto& tensor_name : input_tensor_names_per_segment_[i]) {
      tensor_consumers_count_[tensor_name]++;
    }
  }

  VLOG(1) << "Creating segments runners...";
  for (int i = 0; i < num_segments_; ++i) {
    // `input_tensor_allocator` of the first segment and
    // `output_tensor_allocator` of the last segment are special.
    segments_runners_[i] = absl::make_unique<internal::SegmentRunner>(
        &tensor_consumers_count_, &input_tensor_names_per_segment_[i],
        segments_interpreters_[i], &queues_[i], &queues_[i + 1],
        (i == 0) ? input_tensor_allocator_
                 : intermediate_tensor_allocator_.get(),
        (i == num_segments_ - 1) ? output_tensor_allocator_
                                 : intermediate_tensor_allocator_.get());
  }

  VLOG(1) << "Starting thread for each segment...";
  for (int i = 0; i < num_segments_; ++i) {
    threads_[i] = std::thread(&internal::SegmentRunner::RunInference,
                              segments_runners_[i].get());
  }
}

PipelinedModelRunner::~PipelinedModelRunner() {
  ShutdownPipeline();

  if (!queues_[num_segments_].empty()) {
    LOG(ERROR) << "There are unconsumed output tensors in the pipeline which "
                  "will cause memory leak. Caller is expected to consume all "
                  "the output tensors.";
  }
}

bool PipelinedModelRunner::Push(
    const std::vector<PipelineTensor>& input_tensors) {
  // An empty request signals shutting down the pipeline.
  if (input_tensors.empty()) {
    LOG(INFO) << "Thread: " << std::this_thread::get_id()
              << " receives empty request";
    return ShutdownPipeline();
  }

  auto* interpreter = segments_interpreters_[0];
  CHECK_EQ(interpreter->inputs().size(), input_tensors.size());
  TensorMap managed_input_tensors;
  for (int i = 0; i < input_tensors.size(); ++i) {
    const auto& name = interpreter->input_tensor(i)->name;
    managed_input_tensors.insert({
        name,
        {input_tensors[i], tensor_consumers_count_[name]},
    });
  }

  absl::ReaderMutexLock lock(&mu_);
  if (pipeline_on_) {
    queues_[0].push(managed_input_tensors);
    return true;
  } else {
    LOG(INFO) << "Pipeline was turned off before.";
    return false;
  }
}

bool PipelinedModelRunner::Pop(std::vector<PipelineTensor>* output_tensors) {
  VLOG(1) << "Retrieving output tensors....";

  CHECK_EQ(output_tensors->size(), 0);
  TensorMap managed_output_tensors;

  // Return false if the queue is empty and `StopWaiters()` is called.
  if (!queues_[num_segments_].Wait(&managed_output_tensors)) {
    return false;
  }

  auto* interpreter = segments_interpreters_[num_segments_ - 1];
  CHECK_EQ(managed_output_tensors.size(), interpreter->outputs().size());
  output_tensors->reserve(interpreter->outputs().size());
  for (int i = 0; i < interpreter->outputs().size(); ++i) {
    const auto& name = interpreter->output_tensor(i)->name;
    const auto& managed_output_tensor = managed_output_tensors[name];
    // Sanity check.
    CHECK_EQ(managed_output_tensor.num_consumers, 0);
    output_tensors->push_back(managed_output_tensor.tensor);
  }

  return true;
}

bool PipelinedModelRunner::ShutdownPipeline() {
  absl::MutexLock lock(&mu_);
  if (!pipeline_on_) {
    LOG(INFO) << "Thread: " << std::this_thread::get_id()
              << " Pipeline was turned off before.";
    return false;
  }

  LOG(INFO) << "Thread: " << std::this_thread::get_id()
            << " is shutting down the pipeline...";
  for (int i = 0; i < num_segments_; ++i) {
    queues_[i].StopWaiters();
    // One can only stop queues_[i+1] when threads_[i]'s job is done.
    threads_[i].join();
  }
  queues_[num_segments_].StopWaiters();

  pipeline_on_ = false;
  LOG(INFO) << "Thread: " << std::this_thread::get_id() << " Pipeline is off.";
  return true;
}

std::vector<SegmentStats> PipelinedModelRunner::GetSegmentStats() const {
  std::vector<SegmentStats> result(num_segments_);
  for (int i = 0; i < num_segments_; ++i) {
    result[i] = segments_runners_[i]->stats();
  }
  return result;
}
}  // namespace coral
