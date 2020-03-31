#ifndef EDGETPU_CPP_PIPELINE_INTERNAL_SEGMENT_RUNNER_H_
#define EDGETPU_CPP_PIPELINE_INTERNAL_SEGMENT_RUNNER_H_

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "glog/logging.h"
#include "src/cpp/pipeline/allocator.h"
#include "src/cpp/pipeline/common.h"
#include "src/cpp/pipeline/internal/thread_safe_queue.h"
#include "tensorflow/lite/interpreter.h"

namespace coral {
namespace internal {

// A wrapper on top of PipelineTensor, it keeps track of how many consumers a
// tensor has. This is critical for managing the lifetime of intermediate
// tensors between model segments.
struct ManagedPipelineTensor {
  ManagedPipelineTensor() = default;
  ManagedPipelineTensor(const PipelineTensor& tensor, int num_consumers)
      : tensor(tensor), num_consumers(num_consumers) {}
  PipelineTensor tensor;
  int num_consumers = 0;
};

using TensorMap = std::unordered_map<std::string, ManagedPipelineTensor>;

// Wrapper class that provides API to run inference with a model segment.
//
// Segment runner does not use internal input and output tensor buffers
// allocated by tflite::Interpreter. Instead, it uses input tensor buffers
// allocated by the caller (which will be released by this class using
// `input_tensor_allocator` if applicable) and output tensor buffers allocated
// using `output_tensor_allocator` (by this class).
//
// Note:
//  *) This class assumes interpreter->AllocateTensors() has been called;
class SegmentRunner {
 public:
  SegmentRunner() = default;
  SegmentRunner(
      const std::unordered_map<std::string, int>* tensor_consumers_count,
      const std::unordered_set<std::string>* segment_input_tensor_names,
      tflite::Interpreter* interpreter,
      WaitQueue<internal::TensorMap>* input_queue,
      WaitQueue<internal::TensorMap>* output_queue,
      Allocator* input_tensor_allocator, Allocator* output_tensor_allocator)
      : tensor_consumers_count_(CHECK_NOTNULL(tensor_consumers_count)),
        segment_input_tensor_names_(CHECK_NOTNULL(segment_input_tensor_names)),
        interpreter_(CHECK_NOTNULL(interpreter)),
        input_queue_(CHECK_NOTNULL(input_queue)),
        output_queue_(CHECK_NOTNULL(output_queue)),
        input_tensor_allocator_(CHECK_NOTNULL(input_tensor_allocator)),
        output_tensor_allocator_(CHECK_NOTNULL(output_tensor_allocator)) {}

  // Runs inference until `input_queue_` is stopped and there's no pending
  // requests in the queue.
  void RunInference();

  SegmentStats stats() const {
    absl::ReaderMutexLock lock(&mu_);
    return stats_;
  }

 private:
  // Runs inference once.
  //
  // `input_tensors` are allocated by caller and will be deallocated using
  // `input_tensor_allocator_` if `num_consumers` reaches 0.
  //
  // Returned tensors are allocated by this function using
  // `output_tensor_allocator_`, and it is caller's responsibility to free the
  // memory.
  TensorMap RunInferenceOnce(const TensorMap& input_tensors);

  // Key is tensor name, value is number of consumers for the tensor.
  const std::unordered_map<std::string, int>* tensor_consumers_count_ = nullptr;
  //
  // Note that one can get the same information from `interpreter_`, however,
  // input tensors names are byproducts when caller constructs
  // `tensor_consumers_count_`.
  const std::unordered_set<std::string>* segment_input_tensor_names_ = nullptr;
  tflite::Interpreter* interpreter_ = nullptr;
  WaitQueue<internal::TensorMap>* input_queue_ = nullptr;
  WaitQueue<internal::TensorMap>* output_queue_ = nullptr;
  Allocator* input_tensor_allocator_ = nullptr;
  Allocator* output_tensor_allocator_ = nullptr;

  mutable absl::Mutex mu_;
  SegmentStats stats_ ABSL_GUARDED_BY(mu_);
};
}  // namespace internal
}  // namespace coral

#endif  // EDGETPU_CPP_PIPELINE_INTERNAL_SEGMENT_RUNNER_H_
