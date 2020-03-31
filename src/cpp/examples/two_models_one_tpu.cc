// Example to run two models alternatively using one Edge TPU.
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
// 7. Build and run `two_models_one_tpu`
//
// To reduce variation between different runs, one can disable CPU scaling with
//   sudo cpupower frequency-set --governor performance
#include <algorithm>
#include <chrono>  // NOLINT
#include <iostream>
#include <memory>
#include <string>

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
  const int batch_size = 10;

  std::cout << "Running model: " << bird_model_path
            << " and model: " << plant_model_path << " for " << num_inferences
            << " inferences" << std::endl;

  const auto& start_time = std::chrono::steady_clock::now();
  // Read inputs.
  std::unique_ptr<tflite::FlatBufferModel> bird_model =
      tflite::FlatBufferModel::BuildFromFile(bird_model_path.c_str());
  if (bird_model == nullptr) {
    std::cerr << "Fail to build FlatBufferModel from file: " << bird_model_path
              << std::endl;
    std::abort();
  }
  std::unique_ptr<tflite::FlatBufferModel> plant_model =
      tflite::FlatBufferModel::BuildFromFile(plant_model_path.c_str());
  if (plant_model == nullptr) {
    std::cerr << "Fail to build FlatBufferModel from file: " << plant_model_path
              << std::endl;
    std::abort();
  }

  // This context is shared among multiple models.
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context =
      edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
  std::unique_ptr<tflite::Interpreter> bird_interpreter =
      coral::BuildEdgeTpuInterpreter(*bird_model, edgetpu_context.get());
  std::unique_ptr<tflite::Interpreter> plant_interpreter =
      coral::BuildEdgeTpuInterpreter(*plant_model, edgetpu_context.get());

  std::vector<uint8_t> bird_input = coral::GetInputFromImage(
      bird_image_path, coral::GetInputShape(*bird_interpreter, 0));

  std::vector<uint8_t> plant_input = coral::GetInputFromImage(
      plant_image_path, coral::GetInputShape(*plant_interpreter, 0));

  // Run inference alternately and report timing.
  int num_iterations = (num_inferences + batch_size - 1) / batch_size;
  for (int i = 0; i < num_iterations; ++i) {
    for (int j = 0; j < batch_size; ++j) {
      coral::RunInference(bird_input, bird_interpreter.get());
    }
    for (int j = 0; j < batch_size; ++j) {
      coral::RunInference(plant_input, plant_interpreter.get());
    }
  }
  std::chrono::duration<double> time_span =
      std::chrono::steady_clock::now() - start_time;

  // Print inference result.
  const auto& bird_result =
      coral::RunInference(bird_input, bird_interpreter.get());
  auto it_a = std::max_element(bird_result.begin(), bird_result.end());
  std::cout << "[Bird image analysis] max value index: "
            << std::distance(bird_result.begin(), it_a) << " value: " << *it_a
            << std::endl;
  const auto& plant_result =
      coral::RunInference(plant_input, plant_interpreter.get());
  auto it_b = std::max_element(plant_result.begin(), plant_result.end());
  std::cout << "[Plant image analysis] max value index: "
            << std::distance(plant_result.begin(), it_b) << " value: " << *it_b
            << std::endl;

  std::cout << "Using one Edge TPU, # inferences: " << num_inferences
            << " costs: " << time_span.count() << " seconds." << std::endl;

  return 0;
}
