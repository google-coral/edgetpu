#include "src/cpp/pipeline/internal/segment_runner.h"

#include "absl/synchronization/mutex.h"
#include "glog/logging.h"

namespace coral {
namespace internal {
namespace {
TfLiteFloatArray* TfLiteFloatArrayCopy(const TfLiteFloatArray* src) {
  if (!src) {
    return nullptr;
  }
  TfLiteFloatArray* ret = static_cast<TfLiteFloatArray*>(
      std::malloc(TfLiteFloatArrayGetSizeInBytes(src->size)));
  if (!ret) {
    return nullptr;
  }
  ret->size = src->size;
  std::memcpy(ret->data, src->data, src->size * sizeof(float));
  return ret;
}

// Forces tflite::Interpreter to use external buffer for particular tensor.
void SetExternalTensorBuffer(const char* buffer, std::size_t size_bytes,
                             int tensor_index,
                             tflite::Interpreter* interpreter) {
  const TfLiteTensor* tensor = interpreter->tensor(tensor_index);
  const TfLiteType& type = tensor->type;
  const char* name = tensor->name;
  std::vector<int> dims(tensor->dims->data,
                        tensor->dims->data + tensor->dims->size);
  if (tensor->quantization.type == kTfLiteNoQuantization) {
    // Deal with legacy model with old quantization parameters.
    CHECK_EQ(
        interpreter->SetTensorParametersReadOnly(
            tensor_index, type, name, dims, tensor->params, buffer, size_bytes),
        kTfLiteOk);
  } else {
    // For models with new quantization parameters, deep copy the parameters.
    CHECK(tensor->quantization.type == kTfLiteAffineQuantization);
    CHECK(tensor->quantization.params);
    TfLiteQuantization quant_clone = tensor->quantization;
    const auto* quant_params = reinterpret_cast<TfLiteAffineQuantization*>(
        tensor->quantization.params);
    // |quant_params_clone| will be owned by |quant_clone|, and will be
    // deallocated by std::free(). Therefore std::malloc() is used to allocate
    // its memory here.
    auto* quant_params_clone = reinterpret_cast<TfLiteAffineQuantization*>(
        malloc(sizeof(TfLiteAffineQuantization)));
    quant_params_clone->scale = TfLiteFloatArrayCopy(quant_params->scale);
    CHECK(quant_params_clone->scale);
    quant_params_clone->zero_point =
        TfLiteIntArrayCopy(quant_params->zero_point);
    CHECK(quant_params_clone->zero_point);
    quant_params_clone->quantized_dimension = quant_params->quantized_dimension;
    quant_clone.params = quant_params_clone;
    CHECK(interpreter->SetTensorParametersReadOnly(tensor_index, type, name,
                                                   dims, quant_clone, buffer,
                                                   size_bytes) == kTfLiteOk);
  }

  // Sanity check.
  const auto* tflite_tensor = interpreter->tensor(tensor_index);
  CHECK(tflite_tensor->data.data == buffer)
      << "Tensor is not using the given buffer!";
}
}  // namespace

TensorMap SegmentRunner::RunInferenceOnce(const TensorMap& input_tensors) {
  const auto& start_time = std::chrono::steady_clock::now();

  // Allocate output tensors.
  TensorMap output_tensors;
  for (const auto& output_tensor_index : interpreter_->outputs()) {
    const auto* tflite_tensor = interpreter_->tensor(output_tensor_index);
    PipelineTensor output_tensor;
    output_tensor.type = tflite_tensor->type;
    output_tensor.data.data =
        CHECK_NOTNULL(output_tensor_allocator_->alloc(tflite_tensor->bytes));
    output_tensor.bytes = tflite_tensor->bytes;
    output_tensors.insert(
        {tflite_tensor->name, {output_tensor, /*num_consumers=*/0}});
  }

  // Force tflite interpreter to use external buffers for input tensors.
  for (const auto& tensor_index : interpreter_->inputs()) {
    const auto& it =
        input_tensors.find(interpreter_->tensor(tensor_index)->name);
    CHECK(it != input_tensors.end());
    SetExternalTensorBuffer(
        reinterpret_cast<const char*>(it->second.tensor.data.data),
        it->second.tensor.bytes, tensor_index, interpreter_);
  }

  // Force tflite interpreter to use external buffers for output tensors.
  for (const auto& tensor_index : interpreter_->outputs()) {
    const auto& it =
        output_tensors.find(interpreter_->tensor(tensor_index)->name);
    CHECK(it != output_tensors.end());
    SetExternalTensorBuffer(
        reinterpret_cast<const char*>(it->second.tensor.data.data),
        it->second.tensor.bytes, tensor_index, interpreter_);
  }

  CHECK(interpreter_->Invoke() == kTfLiteOk);

  std::chrono::duration<int64_t, std::nano> time_span =
      std::chrono::steady_clock::now() - start_time;

  {
    absl::MutexLock lock(&mu_);
    stats_.total_time_ns += time_span.count();
    stats_.num_inferences++;
  }

  return output_tensors;
}

void SegmentRunner::RunInference() {
  TensorMap input_tensors;
  while (input_queue_->Wait(&input_tensors)) {
    auto output_tensors = RunInferenceOnce(input_tensors);

    // Set output tensors' consumers count
    for (auto& pair : output_tensors) {
      auto& name = pair.first;
      auto& tensor = pair.second;
      const auto& it = tensor_consumers_count_->find(name);
      tensor.num_consumers =
          (it != tensor_consumers_count_->end()) ? it->second : 0;
    }

    // Reduce consumers count for used input tensors. For tensors that still
    // have consumers, let them flow to the next segment. Otherwise, release the
    // memory if caller provides a valid allocator.
    for (auto& pair : input_tensors) {
      auto& name = pair.first;
      auto& tensor = pair.second;
      // `input_tensors` is unconsumed tensors from previous segments, it can
      // contain tensors that will not be used by this segment.
      if (segment_input_tensor_names_->find(name) !=
          segment_input_tensor_names_->end()) {
        tensor.num_consumers--;
      }
      if (tensor.num_consumers > 0) {
        // Flow to the next segment.
        output_tensors.insert({name, tensor});
      } else {
        // Clean up input tensor.
        VLOG(1) << "Releasing " << name
                << " at addr: " << static_cast<void*>(tensor.tensor.data.data);
        input_tensor_allocator_->free(tensor.tensor.data.data,
                                      tensor.tensor.bytes);
      }
    }

    output_queue_->push(output_tensors);
  }
}

}  // namespace internal
}  // namespace coral
