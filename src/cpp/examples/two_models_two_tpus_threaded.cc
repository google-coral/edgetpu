// Example to run two models with two Edge TPUs using two threads.
// It depends only on tflite and edgetpu.h
//
// Example usage:
// 1. Create directory /tmp/edgetpu_cpp_example
// 2. wget -O /tmp/edgetpu_cpp_example/inat_bird_edgetpu.tflite \
//      http://storage.googleapis.com/cloud-iot-edge-pretrained-models/canned_models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite
// 3. wget -O /tmp/edgetpu_cpp_example/inat_plant_edgetpu.tflite \
//      http://storage.googleapis.com/cloud-iot-edge-pretrained-models/canned_models/mobilenet_v2_1.0_224_inat_plant_quant_edgetpu.tflite
// 4. wget -O /tmp/edgetpu_cpp_example/bird.jpg \
//      https://farm3.staticflickr.com/8008/7523974676_40bbeef7e3_o.jpg
// 5. wget -O /tmp/edgetpu_cpp_example/plant.jpg \
//      https://c2.staticflickr.com/1/62/184682050_db90d84573_o.jpg
// 6. cd /tmp/edgetpu_cpp_example && mogrify -format bmp *.jpg
// 7. Build and run `two_models_two_tpus_threaded`
//
// To reduce variation between different runs, one can disable CPU scaling with
//   sudo cpupower frequency-set --governor performance
#include <algorithm>
#include <chrono>  // NOLINT
#include <iostream>
#include <memory>
#include <string>
#include <thread>  // NOLINT

#include "edgetpu.h"
#include "src/cpp/examples/model_utils.h"
#include "src/cpp/test_utils.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

int main(int argc, char* argv[]) {
  if (argc != 1 && argc != 5) {
    std::cout << argv[0]
              << " <bird_model> <plant_model> <bird_image> <plant_image>"
              << std::endl;
    return 1;
  }

  // Modify the following accordingly to try different models.
  const std::string bird_model_path =
      argc == 5 ? argv[1]
                : coral::GetTempPrefix() +
                      "/edgetpu_cpp_example/inat_bird_edgetpu.tflite";
  const std::string plant_model_path =
      argc == 5 ? argv[2]
                : coral::GetTempPrefix() +
                      "/edgetpu_cpp_example/inat_plant_edgetpu.tflite";
  const std::string bird_image_path =
      argc == 5 ? argv[3]
                : coral::GetTempPrefix() + "/edgetpu_cpp_example/bird.bmp";
  const std::string plant_image_path =
      argc == 5 ? argv[4]
                : coral::GetTempPrefix() + "/edgetpu_cpp_example/plant.bmp";

  const int num_inferences = 2000;

  const auto& available_tpus =
      edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
  if (available_tpus.size() < 2) {
    std::cerr << "This example requires two Edge TPUs to run." << std::endl;
    return 0;
  }

  auto thread_job =
      [num_inferences]  // NOLINT(clang-diagnostic-unused-lambda-capture)
      (const edgetpu::EdgeTpuManager::DeviceEnumerationRecord& tpu,
       const std::string& model_path, const std::string& image_path) {
        const auto& tid = std::this_thread::get_id();
        std::cout << "Thread: " << tid << " Using model: " << model_path
                  << " Running " << num_inferences << " inferences."
                  << std::endl;
        std::unordered_map<std::string, std::string> options = {
            {"Usb.MaxBulkInQueueLength", "8"},
        };
        auto tpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(
            tpu.type, tpu.path, options);
        std::unique_ptr<tflite::FlatBufferModel> model =
            tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
        if (model == nullptr) {
          std::cerr << "Fail to build FlatBufferModel from file: " << model_path
                    << std::endl;
          std::abort();
        }
        std::unique_ptr<tflite::Interpreter> interpreter =
            coral::BuildEdgeTpuInterpreter(*model, tpu_context.get());
        std::cout << "Thread: " << tid << " Interpreter was built."
                  << std::endl;
        std::vector<uint8_t> input = coral::GetInputFromImage(
            image_path, coral::GetInputShape(*interpreter, 0));
        for (int i = 0; i < num_inferences; ++i) {
          coral::RunInference(input, interpreter.get());
        }
        // Print inference result.
        const auto& result = coral::RunInference(input, interpreter.get());
        auto it_a = std::max_element(result.begin(), result.end());
        std::cout << "Thread: " << tid
                  << " printing analysis result. Max value index: "
                  << std::distance(result.begin(), it_a) << " value: " << *it_a
                  << std::endl;
      };

  const auto& start_time = std::chrono::steady_clock::now();
  std::thread bird_model_thread(thread_job, available_tpus[0], bird_model_path,
                                bird_image_path);
  std::thread plant_model_thread(thread_job, available_tpus[1],
                                 plant_model_path, plant_image_path);
  bird_model_thread.join();
  plant_model_thread.join();
  std::chrono::duration<double> time_span =
      std::chrono::steady_clock::now() - start_time;
  std::cout << "Using two Edge TPUs, # inferences: " << num_inferences
            << " costs: " << time_span.count() << " seconds." << std::endl;

  return 0;
}
