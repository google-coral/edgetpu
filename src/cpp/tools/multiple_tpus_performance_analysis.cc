// Tool to do simple performance analysis when using multiple Edge TPU devices.
//
// Basically, it tries to run `num_inferences` inferences with 1, 2, ...,
// [Max # of Edge TPUs] available on host; and record the wall time.
//
// It does this for each model and reports speedup in the end.
//
// To reduce variation between different runs, one can disable CPU scaling with
//   sudo cpupower frequency-set --governor performance

#include <chrono>  // NOLINT
#include <thread>  // NOLINT

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "glog/logging.h"
#include "src/cpp/basic/basic_engine.h"
#include "src/cpp/test_utils.h"

ABSL_FLAG(int, num_requests, 30000, "Number of inference requests to run.");

using EdgeTpuState = coral::EdgeTpuResourceManager::EdgeTpuState;

namespace coral {

// Returns processing wall time in milliseconds.
double ProcessRequests(const std::string& model_name, int num_threads,
                       int num_requests) {
  std::vector<std::thread> workers;
  workers.reserve(num_threads);
  // Divide work among different threads, round up a bit if not divisible.
  int num_requests_per_thread = (num_requests + num_threads - 1) / num_threads;
  auto thread_job = [&]() {
    const auto& tid = std::this_thread::get_id();
    LOG(INFO) << "thread: " << tid
              << " # requests need to process: " << num_requests_per_thread;
    coral::BasicEngine engine(TestDataPath(model_name));
    const auto& input_shape = engine.get_input_tensor_shape();
    const auto& input_tensor =
        coral::GetRandomInput({input_shape[1], input_shape[2], input_shape[3]});
    std::vector<std::vector<float>> results;
    for (int i = 0; i < num_requests_per_thread; ++i) {
      results = engine.RunInference(input_tensor);
    }
    LOG(INFO) << "thread: " << tid << " finished processing requests.";
  };

  const auto& start_time = std::chrono::steady_clock::now();
  for (int i = 0; i < num_threads; ++i) {
    workers.push_back(std::thread(thread_job));
  }
  for (int i = 0; i < num_threads; ++i) {
    workers[i].join();
  }
  std::chrono::duration<double, std::milli> time_span =
      std::chrono::steady_clock::now() - start_time;
  return time_span.count();
}
}  // namespace coral

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const int num_tpus = coral::EdgeTpuResourceManager::GetSingleton()
                           ->ListEdgeTpuPaths(EdgeTpuState::kUnassigned)
                           .size();
  CHECK_GT(num_tpus, 1) << "Need > 1 Edge TPU for the run to be meaningful";

  const std::vector<std::string> models_to_check = {
      "mobilenet_v1_1.0_224_quant_edgetpu.tflite",
      "mobilenet_v2_1.0_224_quant_edgetpu.tflite",
      "mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite",
      "mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite",
      "inception_v1_224_quant_edgetpu.tflite",
      "inception_v2_224_quant_edgetpu.tflite",
      "inception_v3_299_quant_edgetpu.tflite",
      "inception_v4_299_quant_edgetpu.tflite",
  };

  auto print_speedup = [](const std::vector<double>& time_vec) {
    CHECK_GT(time_vec.size(), 1);
    LOG(INFO) << "Single Edge TPU base time " << time_vec[0] << " seconds.";
    for (int i = 1; i < time_vec.size(); ++i) {
      LOG(INFO) << "# TPUs: " << (i + 1)
                << " speedup: " << time_vec[0] / time_vec[i];
    }
  };

  std::map<std::string, std::vector<double>> processing_time_map;
  for (const auto& model_name : models_to_check) {
    auto& time_vec = processing_time_map[model_name];
    time_vec.resize(num_tpus);
    // Run with max number of Edge TPUs first on purpose, otherwise, it can take
    // a long time for user to realize there is not enough Edge TPUs on host.
    for (int i = num_tpus - 1; i >= 0; --i) {
      time_vec[i] = coral::ProcessRequests(model_name,
                                           /*num_threads=*/(i + 1),
                                           absl::GetFlag(FLAGS_num_requests));
      LOG(INFO) << "Model name: " << model_name << " # TPUs: " << (i + 1)
                << " processing time: " << time_vec[i];
    }
    print_speedup(time_vec);
  }

  LOG(INFO) << "===========Summary=============";
  for (const auto& model_name : models_to_check) {
    LOG(INFO) << "----------------------";
    LOG(INFO) << "Model name: " << model_name;
    const auto& time_vec = processing_time_map[model_name];
    print_speedup(time_vec);
  }
  return 0;
}
