// Examples on how to use model pipelining library.
//
// To run this example,
// 1) copy test_data/pipeline folder to /tmp/data
// 2) build and run like,
//     model_pipelining /tmp/data inception_v3_299_quant 3 100
#include <chrono>  // NOLINT
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "edgetpu.h"
#include "src/cpp/pipeline/pipelined_model_runner.h"
#include "src/cpp/pipeline/utils.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

enum class EdgeTpuType {
  kAny,
  kPciOnly,
  kUsbOnly,
};

// Prepares Edge TPU contexts, returns empty vector if there is not enough Edge
// TPUs on the system.
std::vector<std::shared_ptr<edgetpu::EdgeTpuContext>> PrepareEdgeTpuContexts(
    int num_tpus, EdgeTpuType device_type) {
  auto get_available_tpus = [](EdgeTpuType device_type) {
    const auto& all_tpus =
        edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
    if (device_type == EdgeTpuType::kAny) {
      return all_tpus;
    } else {
      std::vector<edgetpu::EdgeTpuManager::DeviceEnumerationRecord> result;

      edgetpu::DeviceType target_type;
      if (device_type == EdgeTpuType::kPciOnly) {
        target_type = edgetpu::DeviceType::kApexPci;
      } else if (device_type == EdgeTpuType::kUsbOnly) {
        target_type = edgetpu::DeviceType::kApexUsb;
      } else {
        std::cerr << "Invalid device type" << std::endl;
        return result;
      }

      for (const auto& tpu : all_tpus) {
        if (tpu.type == target_type) {
          result.push_back(tpu);
        }
      }

      return result;
    }
  };

  const auto& available_tpus = get_available_tpus(device_type);
  if (available_tpus.size() < num_tpus) {
    std::cerr << "Not enough Edge TPU detected, expected: " << num_tpus
              << " actual: " << available_tpus.size();
    return {};
  }

  std::unordered_map<std::string, std::string> options = {
      {"Usb.MaxBulkInQueueLength", "8"},
  };

  std::vector<std::shared_ptr<edgetpu::EdgeTpuContext>> edgetpu_contexts(
      num_tpus);
  for (int i = 0; i < num_tpus; ++i) {
    edgetpu_contexts[i] = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(
        available_tpus[i].type, available_tpus[i].path, options);
    std::cout << "Device " << available_tpus[i].path << " is selected."
              << std::endl;
  }

  return edgetpu_contexts;
}

std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(
    const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* context) {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());

  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder interpreter_builder(model.GetModel(), resolver);
  if (interpreter_builder(&interpreter) != kTfLiteOk) {
    std::cerr << "Error in interpreter initialization." << std::endl;
    return nullptr;
  }

  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, context);
  interpreter->SetNumThreads(1);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors." << std::endl;
    return nullptr;
  }

  return interpreter;
}

std::vector<coral::PipelineTensor> CreateRandomInputTensors(
    const tflite::Interpreter* interpreter, coral::Allocator* allocator) {
  auto fill_random = [](uint8_t* buffer, size_t size) {
    std::default_random_engine generator;
    std::uniform_int_distribution<> distribution(0, UINT8_MAX);
    std::generate(buffer, buffer + size,
                  [&]() { return distribution(generator); });
  };

  std::vector<coral::PipelineTensor> input_tensors;
  for (int input_index : interpreter->inputs()) {
    const auto* input_tensor = interpreter->tensor(input_index);
    coral::PipelineTensor input_buffer;
    input_buffer.data.data = allocator->alloc(input_tensor->bytes);
    input_buffer.bytes = input_tensor->bytes;
    input_buffer.type = input_tensor->type;
    fill_random(static_cast<uint8_t*>(input_buffer.data.data),
                input_buffer.bytes);
    input_tensors.push_back(input_buffer);
  }
  return input_tensors;
}

int main(int argc, char* argv[]) {
  const int kNumArgs = 5;
  if (argc != 1 && argc != kNumArgs) {
    std::cout << " model_pipelining <data_folder> <model_base_name> "
                 "<num_segments> <num_inferences>"
              << std::endl;
    return 1;
  }

  const std::string data_dir = (argc == kNumArgs) ? argv[1] : "/tmp/data/";
  const std::string model_base_name =
      (argc == kNumArgs) ? argv[2] : "inception_v3_299_quant";
  const int num_segments = (argc == kNumArgs) ? std::stoi(argv[3]) : 3;
  const int num_inferences = (argc == kNumArgs) ? std::stoi(argv[4]) : 100;
  std::cout << "data_dir: " << data_dir << std::endl;
  std::cout << "model_base_name: " << model_base_name << std::endl;
  std::cout << "num_segments: " << num_segments << std::endl;
  std::cout << "num_inferences: " << num_inferences << std::endl;

  std::cout << "Preparing Edge TPU contexts" << std::endl;
  auto contexts = PrepareEdgeTpuContexts(num_segments, EdgeTpuType::kAny);
  if (contexts.empty()) {
    return 1;
  }

  std::cout << "Building model pipeline runner" << std::endl;
  std::vector<std::string> model_segment_paths(num_segments);
  for (int i = 0; i < num_segments; ++i) {
    model_segment_paths[i] = data_dir + "/" + model_base_name + "_segment_" +
                             std::to_string(i) + "_of_" +
                             std::to_string(num_segments) + "_edgetpu.tflite";
    std::cout << "model segment: " << model_segment_paths[i] << std::endl;
  }
  std::vector<std::unique_ptr<tflite::Interpreter>> managed_interpreters(
      num_segments);
  std::vector<tflite::Interpreter*> interpreters(num_segments);
  std::vector<std::unique_ptr<tflite::FlatBufferModel>> models(num_segments);
  for (int i = 0; i < num_segments; ++i) {
    models[i] =
        tflite::FlatBufferModel::BuildFromFile(model_segment_paths[i].c_str());
    if (models[i] == nullptr) {
      return 1;
    }
    managed_interpreters[i] =
        BuildEdgeTpuInterpreter(*(models[i]), contexts[i].get());
    if (managed_interpreters[i] == nullptr) {
      return 1;
    }
    interpreters[i] = managed_interpreters[i].get();
  }
  std::unique_ptr<coral::PipelinedModelRunner> runner(
      new coral::PipelinedModelRunner(interpreters));

  std::cout << "Running inference " << num_inferences << " times" << std::endl;
  std::vector<std::vector<coral::PipelineTensor>> input_requests(
      num_inferences);
  for (int i = 0; i < num_inferences; ++i) {
    input_requests[i] = CreateRandomInputTensors(
        interpreters[0], runner->GetInputTensorAllocator());
  }

  auto request_producer = [&runner, &input_requests]() {
    for (const auto& request : input_requests) {
      runner->Push(request);
    }
    runner->Push({});
  };

  auto request_consumer = [&runner]() {
    std::vector<coral::PipelineTensor> output_tensors;
    while (runner->Pop(&output_tensors)) {
      coral::FreeTensors(output_tensors, runner->GetOutputTensorAllocator());
      output_tensors.clear();
    }
    std::cout << "All tensors consumed" << std::endl;
  };

  const auto& start_time = std::chrono::steady_clock::now();
  auto producer = std::thread(request_producer);
  auto consumer = std::thread(request_consumer);
  producer.join();
  consumer.join();
  std::chrono::duration<double, std::milli> time_span =
      std::chrono::steady_clock::now() - start_time;

  std::cout << "Average inference time (in ms): "
            << time_span.count() / num_inferences << std::endl;

  return 0;
}
