#ifndef EDGETPU_CPP_PIPELINE_PIPELINED_MODEL_RUNNER_H_
#define EDGETPU_CPP_PIPELINE_PIPELINED_MODEL_RUNNER_H_

#include <thread>  // NOLINT

#include "absl/synchronization/mutex.h"
#include "src/cpp/pipeline/allocator.h"
#include "src/cpp/pipeline/common.h"
#include "src/cpp/pipeline/internal/segment_runner.h"
#include "src/cpp/pipeline/internal/thread_safe_queue.h"
#include "tensorflow/lite/interpreter.h"

namespace coral {

// Runs inferencing for a segmented model, using a pipeline of Edge TPUs.
// This class assumes each segment has a dedicated Edge TPU, which allows all
// segments to run in parallel and improves throughput.
//
// For example, if you have a pool of requests to process:
//
//    ```
//    auto model_segments_interpreters =
//        ModelSegmentsInterpreters(model_segments_paths);
//    // Caller can set custom allocators for input and output tensors with
//    // `input_tensor_allocator` and `output_tensor_allocator` arguments.
//    auto runner = PipelinedModelRunner(model_segments_interpreters);
//    auto* input_tensor_allocator = runner.GetInputTensorAllocator();
//    auto* output_tensor_allocator = runner.GetOutputTensorAllocator();
//
//    const int total_num_requests = 1000;
//
//    auto request_producer = [&runner, &total_num_requests]() {
//      for (int i = 0; i < total_num_requests; ++i) {
//        // Caller is responsible for allocating input tensors.
//        runner.Push(CreateInputTensors(input_tensor_allocator));
//      }
//    };
//
//    auto result_consumer = [&runner, &total_num_requests]() {
//      for (int i = 0; i < total_num_requests; ++i) {
//        std::vector<Tensor> output_tensors;
//        runner.Pop(&output_tensors);
//        ConsumeOutputTensors(output_tensors);
//        // Caller is responsible for deallocating output tensors.
//        FreeTensors(output_tensor_allocator, output_tensors);
//      }
//    };
//
//    auto producer_thread = std::thread(request_producer);
//    auto consumer_thread = std::thread(result_consumer);
//
//    ```
//
// Or, if you have a stream of requests to process:
//
//    ```
//    auto model_segments_interpreters =
//        ModelSegmentsInterpreters(model_segments_paths);
//    // Caller can set custom allocators for input and output tensors with
//    // `input_tensor_allocator` and `output_tensor_allocator` arguments.
//    auto runner = PipelinedModelRunner(model_segments_interpreters);
//    auto* input_tensor_allocator = runner.GetInputTensorAllocator();
//    auto* output_tensor_allocator = runner.GetOutputTensorAllocator();
//
//    auto request_producer = [&runner]() {
//      while (true) {
//        // Caller is responsible for allocating input tensors.
//        runner.Push(CreateInputTensors(input_tensor_allocator));
//        if (ShouldStop()) {
//          // Pushing special inputs to signal no more inputs will be pushed.
//          runner.Push({});
//          break;
//        }
//      }
//    };
//
//    auto result_consumer = [&runner]() {
//      std::vector<Tensor> output_tensors;
//      while (runner.Pop(&output_tensors)) {
//        ConsumeOutputTensors(output_tensors);
//        // Caller is responsible for deallocating output tensors.
//        FreeTensors(output_tensor_allocator, output_tensors);
//      }
//    };
//
//    auto producer_thread = std::thread(request_producer);
//    auto consumer_thread = std::thread(result_consumer);
//    ```
//
// This class is thread-safe.
class PipelinedModelRunner {
 public:
  // Initializes the PipelinedModelRunner with model segments.
  //
  // @param model_segments_interpreters
  // A vector of pointers to tflite::Interpreter
  // objects, each representing a model segment and unique Edge TPU context.
  // `model_segments_interpreters[0]` should be the first segment interpreter of
  // the model, `model_segments_interpreters[1]` is the second segment, and so
  // on.
  // @param input_tensor_allocator A custom Allocator for input tensors. By
  // default (`nullptr`), it uses an allocator provided by this class.
  // @param output_tensor_allocator A custom Allocator for output tensors. By
  // default (`nullptr`), it uses an allocator provided by this class.
  //
  // **Note:**
  //  * `input_tensor_allocator` is only used to free the input tensors, as
  //     this class assumes that input tensors are allocated by caller.
  //  * `output_tensor_allocator` is only used to allocate output tensors,
  //      as this class assumes that output tensors are freed by caller
  //      after consuming them.
  explicit PipelinedModelRunner(
      const std::vector<tflite::Interpreter*>& model_segments_interpreters,
      Allocator* input_tensor_allocator = nullptr,
      Allocator* output_tensor_allocator = nullptr);

  ~PipelinedModelRunner();

  // Returns the default input tensor allocator (or the allocator given to the
  // constructor).
  Allocator* GetInputTensorAllocator() const { return input_tensor_allocator_; }

  // Returns the default output tensor allocator (or the allocator given to the
  // constructor).
  Allocator* GetOutputTensorAllocator() const {
    return output_tensor_allocator_;
  }

  // Pushes input tensors to be processed by the pipeline.
  //
  // @param input_tensors A vector of input tensors, each wrapped as a
  // PipelineTensor. The order must match Interpreter::inputs() from the
  // first model segment.
  // @return True if successful; false otherwise.
  //
  // **Note:**
  //   *  Caller is responsible for allocating memory for input tensors. By
  //      default, this class will free those tensors when they are consumed.
  //      Caller can set a custom allocator for input tensors if needed.
  //
  //   *  Pushing an empty vector `{}` is allowed, which signals the class that
  //      no more inputs will be added (the function will return false if inputs
  //      were pushed after this special push). This special push allows
  //      Pop()'s consumer to properly drain unconsumed output tensors. See
  //      above example for details.
  bool Push(const std::vector<PipelineTensor>& input_tensors);

  // Gets output tensors from the pipeline.
  //
  // @param output_tensors A pointer to a vector of PipelineTensor objects
  // where outputs should be stored. Returned output tensors order matches
  // Interpreter::outputs() of the last model segment.
  //
  // @return True when output is received, or false when special empty push is
  // given to Push() and there is no more output tensors available.
  //
  // **Note:**
  //   *  Caller is responsible for deallocating memory for output tensors after
  //      consuming the tensors. By default, the output tensors are allocated
  //      using default tensor allocator. Caller can set a custom allocator for
  //      output tensors if needed.
  //
  //   *  Caller will get blocked if there is no output tensors available and no
  //      empty push is received.
  bool Pop(std::vector<PipelineTensor>* output_tensors);

  // Returns performance stats for each segment.
  std::vector<SegmentStats> GetSegmentStats() const;

 private:
  // Returns true if pipeline was shutdown successfully, false if pipeline was
  // shutdown before.
  bool ShutdownPipeline() ABSL_LOCKS_EXCLUDED(mu_);

  std::vector<tflite::Interpreter*> segments_interpreters_;

  const int num_segments_;

  // Queues for input, output, and intermediate tensors.
  // `segments_interpreters_[i]` consumes elements from `queues_[i]` and
  // produces elements to `queues_[i+1]`.
  //
  // size = num_segments_ + 1
  std::vector<internal::WaitQueue<internal::TensorMap>> queues_;

  // Each thread works with one model segment. size = num_segments_.
  std::vector<std::thread> threads_;

  // Records how many consumers each input/intermediate tensor has. This is
  // needed for each segment to decide when to release underlying memory for
  // each input/intermediate tensor.
  std::unordered_map<std::string, int> tensor_consumers_count_;

  // Segment runner is a convenient wrapper that gathers everything that is
  // needed to run one model segment.
  std::vector<std::unique_ptr<internal::SegmentRunner>> segments_runners_;

  // `input_tensor_names_per_segment_[i]` stores input tensors names for the
  // i-th model segment.
  std::vector<std::unordered_set<std::string>> input_tensor_names_per_segment_;

  // Default tensor allocator for input and output tensors if caller does not
  // provide one.
  std::unique_ptr<Allocator> default_allocator_;
  // Tensor allocator for intermediate tensors.
  std::unique_ptr<Allocator> intermediate_tensor_allocator_;
  // Memory allocator for input tensors (of the first model segment).
  Allocator* input_tensor_allocator_ = nullptr;
  // Memory allocator for output tensors (of the last model segment).
  Allocator* output_tensor_allocator_ = nullptr;

  absl::Mutex mu_;
  bool pipeline_on_ ABSL_GUARDED_BY(mu_) = true;
};

}  // namespace coral

#endif  // EDGETPU_CPP_PIPELINE_PIPELINED_MODEL_RUNNER_H_
