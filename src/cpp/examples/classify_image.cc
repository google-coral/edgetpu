// An example to classify image.
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "src/cpp/classification/engine.h"
#include "src/cpp/examples/label_utils.h"
#include "src/cpp/examples/model_utils.h"
#include "src/cpp/test_utils.h"

ABSL_FLAG(std::string, model_path,
          coral::GetTempPrefix() + "/mobilenet_v1_1.0_224_quant_edgetpu.tflite",
          "Path to the tflite model.");

ABSL_FLAG(std::string, image_path, coral::GetTempPrefix() + "/cat.bmp",
          "Path to the image to be classified.");

ABSL_FLAG(std::string, labels_path,
          coral::GetTempPrefix() + "/imagenet_labels.txt",
          "Path to the imagenet labels.");

void ClassifyImage(const std::string& model_path, const std::string& image_path,
                   const std::string& labels_path) {
  // Load the model.
  coral::ClassificationEngine engine(model_path);
  std::vector<int> input_tensor_shape = engine.get_input_tensor_shape();
  // Read the image.
  std::vector<uint8_t> input_tensor = coral::GetInputFromImage(
      image_path,
      {input_tensor_shape[1], input_tensor_shape[2], input_tensor_shape[3]});
  // Read the label file.
  auto labels = coral::ReadLabelFile(labels_path);

  auto results = engine.ClassifyWithInputTensor(input_tensor);
  for (auto result : results) {
    std::cout << "---------------------------" << std::endl;
    std::cout << labels[result.id] << std::endl;
    std::cout << "Score: " << result.score << std::endl;
  }
}

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  ClassifyImage(absl::GetFlag(FLAGS_model_path),
                absl::GetFlag(FLAGS_image_path),
                absl::GetFlag(FLAGS_labels_path));
}
