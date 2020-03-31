// Tool to analyze model pipelining performance.
//
// Run ./model_pipelining_performance_analysis --help to see details on flags.
#include <algorithm>
#include <memory>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "edgetpu.h"
#include "glog/logging.h"
#include "src/cpp/pipeline/common.h"
#include "src/cpp/pipeline/pipelined_model_runner.h"
#include "src/cpp/pipeline/test_utils.h"
#include "src/cpp/pipeline/utils.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

struct IntList {
  std::vector<int> elements;
};

// Returns a textual flag value corresponding to the IntList `list`.
std::string AbslUnparseFlag(const IntList& list) {
  // Let flag module unparse the element type for us.
  return absl::StrJoin(list.elements, ",", [](std::string* out, int element) {
    out->append(absl::UnparseFlag(element));
  });
}

// Parses an IntList from the command line flag value `text`.
// Returns true and sets `*list` on success; returns false and sets `*error` on
// failure.
bool AbslParseFlag(absl::string_view text, IntList* list, std::string* error) {
  // We have to clear the list to overwrite any existing value.
  list->elements.clear();
  // absl::StrSplit("") produces {""}, but we need {} on empty input.
  if (text.empty()) {
    return true;
  }
  for (const auto& part : absl::StrSplit(text, ',')) {
    // Let the flag module parse each element value for us.
    int element;
    if (!absl::ParseFlag(part, &element, error)) {
      return false;
    }
    list->elements.push_back(element);
  }
  return true;
}

enum class EdgeTpuType {
  kAny,
  kPciOnly,
  kUsbOnly,
};

// Parses an EdgeTpuType from the command line flag value. Returns `true` and
// sets `*mode` on success; returns `false` and sets `*error` on failure.
bool AbslParseFlag(absl::string_view text, EdgeTpuType* type,
                   std::string* error) {
  if (text == "any") {
    *type = EdgeTpuType::kAny;
    return true;
  }
  if (text == "pcionly") {
    *type = EdgeTpuType::kPciOnly;
    return true;
  }
  if (text == "usbonly") {
    *type = EdgeTpuType::kUsbOnly;
    return true;
  }
  *error = "unknown value for device_type";
  return false;
}

// Returns a textual flag value corresponding to the EdgeTpuType.
std::string AbslUnparseFlag(EdgeTpuType type) {
  switch (type) {
    case EdgeTpuType::kAny:
      return "any";
    case EdgeTpuType::kPciOnly:
      return "pcionly";
    case EdgeTpuType::kUsbOnly:
      return "usbonly";
    default:
      return absl::StrCat(type);
  }
}

ABSL_FLAG(std::string, data_dir, "/tmp/models/",
          "Models location prefix, this tool assumes data_dir has a flat "
          "layout, i.e. there is no subfolders.");

ABSL_FLAG(std::vector<std::string>, model_list,
          std::vector<std::string>({"inception_v3_299_quant",
                                    "inception_v4_299_quant"}),
          "Comma separated list of model names (without _edgetpu.tflite "
          "suffix) to get performance metric for.");

ABSL_FLAG(IntList, num_segments_list, {std::vector<int>({1, 2, 3, 4})},
          "Comma separated list that specifies number of segments to check for "
          "performance.");

ABSL_FLAG(int, num_inferences, 100, "Number of inferences to run each model.");

ABSL_FLAG(
    EdgeTpuType, device_type, EdgeTpuType::kAny,
    "Type of Edge TPU device to use, values: `pcionly`, `usbonly`, `any`.");

namespace coral {

using edgetpu::EdgeTpuContext;

// num_segments, latency (in ns) pair.
using PerfStats = std::pair<int, int64_t>;

std::vector<std::shared_ptr<EdgeTpuContext>> PrepareEdgeTpuContexts(
    int num_tpus, EdgeTpuType device_type) {
  auto get_available_tpus = [](EdgeTpuType device_type) {
    const auto& all_tpus =
        edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
    if (device_type == EdgeTpuType::kAny) {
      return all_tpus;
    } else {
      edgetpu::DeviceType target_type;
      if (device_type == EdgeTpuType::kPciOnly) {
        target_type = edgetpu::DeviceType::kApexPci;
      } else if (device_type == EdgeTpuType::kUsbOnly) {
        target_type = edgetpu::DeviceType::kApexUsb;
      } else {
        LOG(FATAL) << "Invalid device type";
      }
      std::vector<edgetpu::EdgeTpuManager::DeviceEnumerationRecord> result;
      for (const auto& tpu : all_tpus) {
        if (tpu.type == target_type) {
          result.push_back(tpu);
        }
      }
      return result;
    }
  };
  const auto& available_tpus = get_available_tpus(device_type);
  CHECK_GE(available_tpus.size(), num_tpus);

  std::unordered_map<std::string, std::string> options = {
      {"Usb.MaxBulkInQueueLength", "8"},
  };
  std::vector<std::shared_ptr<EdgeTpuContext>> edgetpu_contexts(num_tpus);
  for (int i = 0; i < num_tpus; ++i) {
    edgetpu_contexts[i] = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(
        available_tpus[i].type, available_tpus[i].path, options);
    LOG(INFO) << "Device " << available_tpus[i].path << " is selected.";
  }

  return edgetpu_contexts;
}

PerfStats BenchmarkPartitionedModel(
    const std::vector<std::string>& model_segments_paths,
    const std::vector<std::shared_ptr<EdgeTpuContext>>* edgetpu_contexts,
    int num_inferences) {
  CHECK_LE(model_segments_paths.size(), edgetpu_contexts->size());
  const int num_segments = model_segments_paths.size();
  std::vector<std::unique_ptr<tflite::Interpreter>> managed_interpreters(
      num_segments);
  std::vector<tflite::Interpreter*> interpreters(num_segments);
  std::vector<std::unique_ptr<tflite::FlatBufferModel>> models(num_segments);
  for (int i = 0; i < num_segments; ++i) {
    models[i] =
        tflite::FlatBufferModel::BuildFromFile(model_segments_paths[i].c_str());
    managed_interpreters[i] =
        coral::CreateInterpreter(*(models[i]), (*edgetpu_contexts)[i].get());
    interpreters[i] = managed_interpreters[i].get();
  }
  auto runner = absl::make_unique<PipelinedModelRunner>(interpreters);

  // Generating input tensors can be quite time consuming, pulling them out to
  // avoid polluting measurement of pipelining latency.
  std::vector<std::vector<PipelineTensor>> input_requests(num_inferences);
  for (int i = 0; i < num_inferences; ++i) {
    input_requests[i] = CreateRandomInputTensors(
        interpreters[0], runner->GetInputTensorAllocator());
  }

  auto request_producer = [&runner, &input_requests]() {
    const auto& start_time = std::chrono::steady_clock::now();
    const auto& num_inferences = input_requests.size();
    for (int i = 0; i < num_inferences; ++i) {
      runner->Push(input_requests[i]);
    }
    runner->Push({});
    std::chrono::duration<int64_t, std::nano> time_span =
        std::chrono::steady_clock::now() - start_time;
    LOG(INFO) << "Producer thread per request latency (in ns): "
              << time_span.count() / num_inferences;
  };

  auto request_consumer = [&runner, &num_inferences]() {
    const auto& start_time = std::chrono::steady_clock::now();
    std::vector<PipelineTensor> output_tensors;
    while (runner->Pop(&output_tensors)) {
      FreeTensors(output_tensors, runner->GetOutputTensorAllocator());
      output_tensors.clear();
    }
    LOG(INFO) << "All tensors consumed";
    std::chrono::duration<int64_t, std::nano> time_span =
        std::chrono::steady_clock::now() - start_time;
    LOG(INFO) << "Consumer thread per request latency (in ns): "
              << time_span.count() / num_inferences;
  };

  const auto& start_time = std::chrono::steady_clock::now();
  auto producer = std::thread(request_producer);
  auto consumer = std::thread(request_consumer);
  producer.join();
  consumer.join();
  std::chrono::duration<int64_t, std::nano> time_span =
      std::chrono::steady_clock::now() - start_time;
  return {num_segments, time_span.count() / num_inferences};
}

}  // namespace coral

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  const auto& data_dir = absl::GetFlag(FLAGS_data_dir);
  const auto& model_list = absl::GetFlag(FLAGS_model_list);
  const auto& num_segments_list = absl::GetFlag(FLAGS_num_segments_list);
  const auto& num_inferences = absl::GetFlag(FLAGS_num_inferences);
  const auto& device_type = absl::GetFlag(FLAGS_device_type);

  LOG(INFO) << "data_dir: " << data_dir;
  LOG(INFO) << "list of models: " << absl::StrJoin(model_list, "\n");
  LOG(INFO) << "num_segments_list: " << AbslUnparseFlag(num_segments_list);
  LOG(INFO) << "num_inferences: " << num_inferences;
  LOG(INFO) << "device_type: " << AbslUnparseFlag(device_type);

  const int max_num_segments = *std::max_element(
      num_segments_list.elements.begin(), num_segments_list.elements.end());
  auto edgetpu_contexts =
      coral::PrepareEdgeTpuContexts(max_num_segments, device_type);

  // Benchmark all model_list and num_segments_list combinations.
  std::unordered_map<std::string, std::vector<coral::PerfStats>> stats_map;
  for (const auto& model_name : model_list) {
    for (const auto& num_segments : num_segments_list.elements) {
      std::vector<std::string> model_segments_paths;
      if (num_segments == 1) {
        model_segments_paths = {data_dir + model_name + "_edgetpu.tflite"};
      } else {
        model_segments_paths =
            coral::SegmentsNames(data_dir + model_name, num_segments);
      }

      const auto& stats = coral::BenchmarkPartitionedModel(
          model_segments_paths, &edgetpu_contexts, num_inferences);
      LOG(INFO) << "Model name: " << model_name
                << " num_segments: " << stats.first
                << " latency (in ns): " << stats.second;

      stats_map[model_name].push_back(stats);
    }
  }

  LOG(INFO) << "========Summary=========";
  for (const auto& model_name : model_list) {
    LOG(INFO) << "Model: " << model_name;
    const auto& baseline = stats_map[model_name][0];
    for (const auto& pair : stats_map[model_name]) {
      LOG(INFO) << "    num_segments: " << pair.first
                << " latency (in ns): " << pair.second << " speedup: "
                << static_cast<float>(baseline.second) / pair.second;
    }
  }

  return 0;
}
